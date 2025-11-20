# dbsi_toolbox/deep_learning.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import sys
from typing import Dict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DBSI_PhysicsDecoder(nn.Module):
    def __init__(self, bvals: np.ndarray, bvecs: np.ndarray, n_iso_bases: int = 20):
        super().__init__()
        self.register_buffer('bvals', torch.tensor(bvals, dtype=torch.float32))
        self.register_buffer('bvecs', torch.tensor(bvecs, dtype=torch.float32))
        
        self.iso_grid_np = np.linspace(0, 3.0e-3, n_iso_bases)
        self.register_buffer('iso_diffusivities', torch.tensor(self.iso_grid_np, dtype=torch.float32))
        self.n_iso = n_iso_bases

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        f_iso_weights = params[:, :self.n_iso]
        f_fiber       = params[:, self.n_iso]
        theta         = params[:, self.n_iso + 1]
        phi           = params[:, self.n_iso + 2]
        d_ax          = params[:, self.n_iso + 3]
        d_rad         = params[:, self.n_iso + 4]

        fiber_dir = torch.stack([
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta)
        ], dim=1)
        cos_angle = torch.matmul(fiber_dir, self.bvecs.T)
        d_app_fiber = d_rad.unsqueeze(1) + (d_ax.unsqueeze(1) - d_rad.unsqueeze(1)) * (cos_angle ** 2)
        signal_fiber = f_fiber.unsqueeze(1) * torch.exp(-self.bvals.unsqueeze(0) * d_app_fiber)

        basis_iso = torch.exp(-torch.ger(self.bvals, self.iso_diffusivities))
        signal_iso = torch.matmul(f_iso_weights, basis_iso.T)

        return signal_fiber + signal_iso


class DBSI_RegularizedMLP(nn.Module):
    """
    Final Tuned Architecture (Fiber Boost).
    Relaxed constraints on Fiber Radial Diffusivity (up to 1.2) 
    and Positive Bias for Fiber Fraction.
    """
    def __init__(self, n_input_meas: int, n_iso_bases: int = 20, dropout_rate: float = 0.05):
        super().__init__()
        self.n_iso = n_iso_bases
        
        grid = np.linspace(0, 3.0e-3, n_iso_bases)
        self.idx_res = torch.tensor(np.where(grid <= 0.3e-3)[0], dtype=torch.long)
        self.idx_hin = torch.tensor(np.where((grid > 0.3e-3) & (grid <= 2.0e-3))[0], dtype=torch.long)
        self.idx_wat = torch.tensor(np.where(grid > 2.0e-3)[0], dtype=torch.long)
        
        self.backbone = nn.Sequential(
            nn.Linear(n_input_meas, 256),
            nn.LayerNorm(256), nn.ELU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 256),
            nn.LayerNorm(256), nn.ELU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.LayerNorm(128), nn.ELU()
        )
        
        self.head_fractions = nn.Linear(128, 4)
        self.head_iso_micro = nn.Linear(128, n_iso_bases)
        self.head_geom = nn.Linear(128, 4)
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
        # TUNING: Fiber Boost
        # [Res, Hin, Wat, Fib]
        with torch.no_grad():
            self.head_fractions.bias[0] = 0.0  # Restricted: Neutral
            self.head_fractions.bias[1] = 0.0  # Hindered: Neutral (was +0.5, removing help)
            self.head_fractions.bias[2] = 0.0  # Water: Neutral
            self.head_fractions.bias[3] = 0.5  # Fiber: POSITIVE BOOST (+0.5)
            # This forces the network to prioritize Fiber fitting initially.

    def forward(self, x):
        feat = self.backbone(x)
        
        probs = self.softmax(self.head_fractions(feat))
        f_res_macro = probs[:, 0:1]
        f_hin_macro = probs[:, 1:2]
        f_wat_macro = probs[:, 2:3]
        f_fiber     = probs[:, 3]
        
        micro_logits = self.head_iso_micro(feat)
        
        res_dist = self.softmax(micro_logits[:, self.idx_res])
        w_res = res_dist * f_res_macro
        
        hin_dist = self.softmax(micro_logits[:, self.idx_hin])
        w_hin = hin_dist * f_hin_macro
        
        wat_dist = self.softmax(micro_logits[:, self.idx_wat])
        w_wat = wat_dist * f_wat_macro
        
        f_iso_weights = torch.cat([w_res, w_hin, w_wat], dim=1)
        
        geom = self.head_geom(feat)
        theta = self.sigmoid(geom[:, 0]) * np.pi
        phi   = (self.sigmoid(geom[:, 1]) - 0.5) * 2 * np.pi
        
        # TUNING: Diffusivity Constraints
        # D_AX: [1.0, 3.0]
        d_ax  = self.sigmoid(geom[:, 2]) * 2.0e-3 + 1.0e-3 
        
        # D_RAD: [0.0, 1.2] (Relaxed from 0.9)
        # We allow fibers to be slightly "fatter" (up to 1.2) so they aren't
        # misclassified as Hindered/Water.
        d_rad = self.sigmoid(geom[:, 3]) * 1.2e-3           
        
        d_ax  = torch.max(d_ax, d_rad + 1e-6)

        return torch.cat([
            f_iso_weights, 
            f_fiber.unsqueeze(1), 
            theta.unsqueeze(1), 
            phi.unsqueeze(1), 
            d_ax.unsqueeze(1), 
            d_rad.unsqueeze(1)
        ], dim=1)

class DBSI_DeepSolver:
    def __init__(self, n_iso_bases: int = 20, epochs: int = 500, batch_size: int = 4096, learning_rate: float = 1e-3, noise_injection_level: float = 0.03):
        self.n_iso_bases = n_iso_bases
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.noise_level = noise_injection_level
        
    def fit_volume(self, volume: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
        print(f"[DeepSolver] Strategy: Fiber Boost (+Bias, Relaxed Drad) + L2 Regularization")
        
        X_vol, Y_vol, Z_vol, N_meas = volume.shape
        valid_signals = volume[mask]
        
        b0_idx = np.where(bvals < 50)[0]
        if len(b0_idx) > 0:
            s0 = np.mean(valid_signals[:, b0_idx], axis=1, keepdims=True)
            s0[s0 < 1e-4] = 1.0 
            valid_signals = valid_signals / s0
            valid_signals = np.clip(valid_signals, 0, 1.5)
        
        dataset = TensorDataset(torch.tensor(valid_signals, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        encoder = DBSI_RegularizedMLP(N_meas, self.n_iso_bases).to(DEVICE)
        decoder = DBSI_PhysicsDecoder(bvals, bvecs, self.n_iso_bases).to(DEVICE)
        optimizer = optim.AdamW(encoder.parameters(), lr=self.lr)
        loss_mse = nn.MSELoss()
        
        encoder.train()
        current_noise = self.noise_level
        pbar = tqdm(range(self.epochs), desc="DL Optimization", unit="epoch", file=sys.stdout)
        
        for epoch in pbar:
            batch_loss = 0.0
            if epoch > 300: current_noise = self.noise_level * 0.5
            if epoch > 450: current_noise = 0.0
            
            for batch in dataloader:
                clean = batch[0].to(DEVICE)
                noise = torch.randn_like(clean) * current_noise
                
                optimizer.zero_grad()
                preds = encoder(clean + noise)
                
                recon = decoder(preds)
                l_fit = loss_mse(recon, clean)
                
                # L2 Regularization
                all_weights = preds[:, :self.n_iso_bases+1]
                l_reg = torch.mean(all_weights ** 2)
                
                loss = l_fit + (0.005 * l_reg) 
                
                loss.backward()
                optimizer.step()
                batch_loss += l_fit.item()
            
            pbar.set_postfix({"MSE": f"{batch_loss/len(dataloader):.6f}"})
            
        print("Inference...")
        encoder.eval()
        full_loader = DataLoader(dataset, batch_size=4096, shuffle=False)
        res = []
        with torch.no_grad():
            for batch in full_loader:
                res.append(encoder(batch[0].to(DEVICE)).cpu().numpy())
        return self._pack_results(np.concatenate(res, axis=0), X_vol, Y_vol, Z_vol, mask)

    def _pack_results(self, flat_results, X, Y, Z, mask):
        def to_3d(flat_arr):
            vol = np.zeros((X, Y, Z), dtype=np.float32)
            vol[mask] = flat_arr
            return vol
            
        n = self.n_iso_bases
        iso_w = flat_results[:, :n]
        f_fib = flat_results[:, n]
        d_ax  = flat_results[:, n+3]
        d_rad = flat_results[:, n+4]
        
        grid = np.linspace(0, 3.0e-3, n)
        mask_res = grid <= 0.3e-3
        mask_hin = (grid > 0.3e-3) & (grid <= 2.0e-3)
        mask_wat = grid > 2.0e-3
        
        theta = flat_results[:, n+1]
        phi   = flat_results[:, n+2]
        dx = np.sin(theta)*np.cos(phi)
        dy = np.sin(theta)*np.sin(phi)
        dz = np.cos(theta)
        
        return {
            'restricted_fraction': to_3d(np.sum(iso_w[:, mask_res], axis=1)),
            'hindered_fraction':   to_3d(np.sum(iso_w[:, mask_hin], axis=1)),
            'water_fraction':      to_3d(np.sum(iso_w[:, mask_wat], axis=1)),
            'fiber_fraction':      to_3d(f_fib),
            'axial_diffusivity':   to_3d(d_ax),
            'radial_diffusivity':  to_3d(d_rad),
            'fiber_dir_x':         to_3d(dx),
            'fiber_dir_y':         to_3d(dy),
            'fiber_dir_z':         to_3d(dz)
        }