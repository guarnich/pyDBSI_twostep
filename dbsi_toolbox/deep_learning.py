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
        
        # Grid standard
        self.iso_grid_np = np.linspace(0, 3.0e-3, n_iso_bases)
        self.register_buffer('iso_diffusivities', torch.tensor(self.iso_grid_np, dtype=torch.float32))
        self.n_iso = n_iso_bases

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        # Unpack params
        # [0..N-1] iso_weights
        # [N] f_fiber
        # [N+1..] geometry
        f_iso_weights = params[:, :self.n_iso]
        f_fiber       = params[:, self.n_iso]
        theta         = params[:, self.n_iso + 1]
        phi           = params[:, self.n_iso + 2]
        d_ax          = params[:, self.n_iso + 3]
        d_rad         = params[:, self.n_iso + 4]

        # Anisotropic
        fiber_dir = torch.stack([
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta)
        ], dim=1)
        cos_angle = torch.matmul(fiber_dir, self.bvecs.T)
        d_app_fiber = d_rad.unsqueeze(1) + (d_ax.unsqueeze(1) - d_rad.unsqueeze(1)) * (cos_angle ** 2)
        signal_fiber = f_fiber.unsqueeze(1) * torch.exp(-self.bvals.unsqueeze(0) * d_app_fiber)

        # Isotropic
        basis_iso = torch.exp(-torch.ger(self.bvals, self.iso_diffusivities))
        signal_iso = torch.matmul(f_iso_weights, basis_iso.T)

        return signal_fiber + signal_iso


class DBSI_RegularizedMLP(nn.Module):
    """
    Macro-Compartment Architecture.
    Predicts aggregate fractions (Restricted/Hindered/Water) explicitly
    before distributing them to the spectral basis.
    """
    def __init__(self, n_input_meas: int, n_iso_bases: int = 20, dropout_rate: float = 0.1):
        super().__init__()
        self.n_iso = n_iso_bases
        
        # Define masks for the spectral grid logic (0-3.0 um2/ms)
        grid = np.linspace(0, 3.0e-3, n_iso_bases)
        self.idx_res = torch.tensor(np.where(grid <= 0.3e-3)[0], dtype=torch.long)
        self.idx_hin = torch.tensor(np.where((grid > 0.3e-3) & (grid <= 2.0e-3))[0], dtype=torch.long)
        self.idx_wat = torch.tensor(np.where(grid > 2.0e-3)[0], dtype=torch.long)
        
        # Main Backbone
        self.backbone = nn.Sequential(
            nn.Linear(n_input_meas, 256),
            nn.LayerNorm(256), nn.ELU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 256),
            nn.LayerNorm(256), nn.ELU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.LayerNorm(128), nn.ELU()
        )
        
        # --- HEAD 1: Fiber Fraction (1 neuron) ---
        self.head_fiber = nn.Linear(128, 1)
        
        # --- HEAD 2: Isotropic MACRO Distribution (3 neurons: Res, Hin, Wat) ---
        self.head_iso_macro = nn.Linear(128, 3)
        
        # --- HEAD 3: Isotropic MICRO Distribution (N neurons) ---
        # Determines the shape WITHIN the compartments
        self.head_iso_micro = nn.Linear(128, n_iso_bases)
        
        # --- HEAD 4: Geometry (4 neurons) ---
        self.head_geom = nn.Linear(128, 4)
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        feat = self.backbone(x)
        
        # 1. Fiber Fraction
        f_fiber = self.sigmoid(self.head_fiber(feat)).squeeze(1)
        
        # 2. Isotropic Macro Fractions (Sum to 1)
        # output: [prob_res, prob_hin, prob_wat]
        iso_macro_probs = self.softmax(self.head_iso_macro(feat))
        
        # 3. Total Isotropic Signal available
        f_iso_total = 1.0 - f_fiber
        
        # 4. Isotropic Micro Shapes (Logits)
        micro_logits = self.head_iso_micro(feat)
        
        # --- COMPARTMENT RECONSTRUCTION ---
        # We construct the full spectrum vector by masking and scaling
        
        # Normalize shape within Restricted group
        res_logits = micro_logits[:, self.idx_res]
        res_dist = self.softmax(res_logits) 
        # Scale by Macro Probability and Total Iso
        w_res = res_dist * iso_macro_probs[:, 0:1] * f_iso_total.unsqueeze(1)
        
        # Normalize shape within Hindered group
        hin_logits = micro_logits[:, self.idx_hin]
        hin_dist = self.softmax(hin_logits)
        w_hin = hin_dist * iso_macro_probs[:, 1:2] * f_iso_total.unsqueeze(1)
        
        # Normalize shape within Water group
        wat_logits = micro_logits[:, self.idx_wat]
        wat_dist = self.softmax(wat_logits)
        w_wat = wat_dist * iso_macro_probs[:, 2:3] * f_iso_total.unsqueeze(1)
        
        # Combine into full vector of size n_iso
        # Create empty container
        f_iso_weights = torch.zeros(x.shape[0], self.n_iso, device=x.device)
        # Scatter weights back (inefficient but clear)
        # Better: concat if indices are contiguous (they are with linspace)
        f_iso_weights = torch.cat([w_res, w_hin, w_wat], dim=1)
        
        # 5. Geometry
        geom = self.head_geom(feat)
        theta = self.sigmoid(geom[:, 0]) * np.pi
        phi   = (self.sigmoid(geom[:, 1]) - 0.5) * 2 * np.pi
        d_ax  = self.sigmoid(geom[:, 2]) * 2.5e-3 + 0.5e-3
        d_rad = self.sigmoid(geom[:, 3]) * 1.5e-3
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
    def __init__(self, n_iso_bases: int = 20, epochs: int = 300, batch_size: int = 2048, learning_rate: float = 1e-3, noise_injection_level: float = 0.03):
        self.n_iso_bases = n_iso_bases
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.noise_level = noise_injection_level
        
    def fit_volume(self, volume: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
        print(f"[DeepSolver] Strategy: Grouped Architecture + L2 Ridge Regularization")
        
        # Data Prep
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
        
        # Models
        encoder = DBSI_RegularizedMLP(N_meas, self.n_iso_bases).to(DEVICE)
        decoder = DBSI_PhysicsDecoder(bvals, bvecs, self.n_iso_bases).to(DEVICE)
        optimizer = optim.AdamW(encoder.parameters(), lr=self.lr)
        loss_mse = nn.MSELoss()
        
        # Training
        encoder.train()
        current_noise = self.noise_level
        pbar = tqdm(range(self.epochs), desc="DL Optimization", unit="epoch", file=sys.stdout)
        
        for epoch in pbar:
            batch_loss = 0.0
            if epoch > self.epochs // 2: current_noise = self.noise_level * 0.5
            if epoch > self.epochs * 0.8: current_noise = 0.0
            
            for batch in dataloader:
                clean = batch[0].to(DEVICE)
                noise = torch.randn_like(clean) * current_noise
                
                optimizer.zero_grad()
                preds = encoder(clean + noise)
                
                # Reconstruction Loss
                recon = decoder(preds)
                l_fit = loss_mse(recon, clean)
                
                # L2 Regularization on Spectrum (Ridge)
                # Instead of forcing zeros (L1/Entropy), we force "smoothness" (L2).
                # This prevents the network from zeroing out Restricted/Water 
                # just because Hindered is dominant.
                iso_weights = preds[:, :self.n_iso_bases]
                l_reg = torch.mean(iso_weights ** 2)
                
                loss = l_fit + (0.01 * l_reg) # 0.01 weight for L2
                
                loss.backward()
                optimizer.step()
                batch_loss += l_fit.item()
            
            pbar.set_postfix({"MSE": f"{batch_loss/len(dataloader):.6f}"})
            
        # Inference
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
        
        # Direction extraction
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