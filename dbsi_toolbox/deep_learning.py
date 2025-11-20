# dbsi_toolbox/deep_learning.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import sys
from typing import Dict

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DBSI_PhysicsDecoder(nn.Module):
    """
    Physics-Informed Decoder (Non-trainable).
    Implements the exact DBSI equation.
    """
    def __init__(self, bvals: np.ndarray, bvecs: np.ndarray, n_iso_bases: int = 50):
        super().__init__()
        
        self.register_buffer('bvals', torch.tensor(bvals, dtype=torch.float32))
        self.register_buffer('bvecs', torch.tensor(bvecs, dtype=torch.float32))
        
        # Standard DBSI grid [0, 3.0] um^2/ms
        iso_grid = torch.linspace(0, 3.0e-3, n_iso_bases)
        self.register_buffer('iso_diffusivities', iso_grid)
        
        self.n_iso = n_iso_bases

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        f_iso_weights = params[:, :self.n_iso]
        f_fiber       = params[:, self.n_iso]
        theta         = params[:, self.n_iso + 1]
        phi           = params[:, self.n_iso + 2]
        d_ax          = params[:, self.n_iso + 3]
        d_rad         = params[:, self.n_iso + 4]

        # 1. Anisotropic Component (Fibers)
        fiber_dir = torch.stack([
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta)
        ], dim=1)

        cos_angle = torch.matmul(fiber_dir, self.bvecs.T)
        d_app_fiber = d_rad.unsqueeze(1) + (d_ax.unsqueeze(1) - d_rad.unsqueeze(1)) * (cos_angle ** 2)
        signal_fiber = f_fiber.unsqueeze(1) * torch.exp(-self.bvals.unsqueeze(0) * d_app_fiber)

        # 2. Isotropic Component (Spectrum)
        basis_iso = torch.exp(-torch.ger(self.bvals, self.iso_diffusivities))
        signal_iso = torch.matmul(f_iso_weights, basis_iso.T)

        return signal_fiber + signal_iso


class DBSI_RegularizedMLP(nn.Module):
    """
    MLP Encoder with Split Fraction Architecture and Custom Initialization.
    """
    def __init__(self, n_input_meas: int, n_iso_bases: int = 50, dropout_rate: float = 0.1):
        super().__init__()
        self.n_iso = n_iso_bases
        self.n_output_features = 1 + n_iso_bases + 2 + 2 
        
        self.net = nn.Sequential(
            nn.Linear(n_input_meas, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, self.n_output_features)
        )
        
        self.sigmoid = nn.Sigmoid()
        self.softmax_iso = nn.Softmax(dim=1)
        
        # --- CUSTOM INITIALIZATION ---
        # Important: Force the network to start with LOW fiber fraction
        # so it learns to use the isotropic spectrum (Hindered/Water) first.
        self._init_weights()

    def _init_weights(self):
        # Initialize the last layer
        last_layer = self.net[-1]
        nn.init.xavier_uniform_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)
        
        # Manually set the bias for the Fiber Fraction neuron (Index 0) to a negative value
        # Sigmoid(-2.0) ~= 0.12 (Starts with 12% fiber, forcing it to find Isotropic)
        with torch.no_grad():
            last_layer.bias[0] = -2.0 

    def forward(self, x):
        raw = self.net(x)
        
        # 1. Fiber Fraction (0 to 1)
        f_fiber_raw = raw[:, 0]
        f_fiber = self.sigmoid(f_fiber_raw)
        
        # 2. Isotropic Distribution (Sums to 1)
        iso_logits = raw[:, 1:self.n_iso + 1]
        iso_dist = self.softmax_iso(iso_logits)
        
        # 3. Scale: f_iso = iso_dist * (1 - f_fiber)
        f_iso_total = 1.0 - f_fiber
        f_iso_weights = iso_dist * f_iso_total.unsqueeze(1)
        
        # 4. Geometry
        idx_geom = self.n_iso + 1
        theta = self.sigmoid(raw[:, idx_geom]) * np.pi
        phi   = (self.sigmoid(raw[:, idx_geom + 1]) - 0.5) * 2 * np.pi
        
        # Diffusivities (Physiological Constraints)
        d_ax = self.sigmoid(raw[:, idx_geom + 2]) * 2.5e-3 + 0.5e-3 # Min 0.5
        d_rad = self.sigmoid(raw[:, idx_geom + 3]) * 1.5e-3         # Max 1.5
        d_ax = torch.max(d_ax, d_rad + 1e-6)

        return torch.cat([
            f_iso_weights, 
            f_fiber.unsqueeze(1), 
            theta.unsqueeze(1), 
            phi.unsqueeze(1), 
            d_ax.unsqueeze(1), 
            d_rad.unsqueeze(1)
        ], dim=1)


class DBSI_DeepSolver:
    def __init__(self, 
                 n_iso_bases: int = 20,  # Fewer bases = easier for Hindered to emerge
                 epochs: int = 300,      
                 batch_size: int = 2048, 
                 learning_rate: float = 5e-4, # Slower LR for stability
                 noise_injection_level: float = 0.03): 
        
        self.n_iso_bases = n_iso_bases
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.noise_level = noise_injection_level
        
    def fit_volume(self, volume: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
        
        print(f"[DeepSolver] Training with Entropy Regularization (Fixing Hindered collapse)...")
        
        # 1. Data Prep
        X_vol, Y_vol, Z_vol, N_meas = volume.shape
        valid_signals = volume[mask]
        
        # S0 Normalization
        b0_idx = np.where(bvals < 50)[0]
        if len(b0_idx) > 0:
            s0 = np.mean(valid_signals[:, b0_idx], axis=1, keepdims=True)
            s0[s0 < 1e-4] = 1.0 
            valid_signals = valid_signals / s0
            valid_signals = np.clip(valid_signals, 0, 1.5)
        
        dataset = TensorDataset(torch.tensor(valid_signals, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        # 2. Init
        encoder = DBSI_RegularizedMLP(N_meas, self.n_iso_bases, dropout_rate=0.05).to(DEVICE)
        decoder = DBSI_PhysicsDecoder(bvals, bvecs, self.n_iso_bases).to(DEVICE)
        optimizer = optim.AdamW(encoder.parameters(), lr=self.lr, weight_decay=1e-4)
        recon_loss_fn = nn.MSELoss() 
        
        # 3. Training
        encoder.train()
        current_noise = self.noise_level
        
        pbar = tqdm(range(self.epochs), desc="DBSI-DL Optimization", unit="epoch", file=sys.stdout)
        
        for epoch in pbar:
            epoch_loss = 0.0
            
            # Noise scheduling
            if epoch > 100: current_noise = self.noise_level * 0.5
            if epoch > 200: current_noise = 0.0
            
            for batch in dataloader:
                clean_signal = batch[0].to(DEVICE)
                noise = torch.randn_like(clean_signal) * current_noise
                
                optimizer.zero_grad()
                
                # Forward
                params_pred = encoder(clean_signal + noise)
                
                # Extract weights for regularization
                iso_weights = params_pred[:, :self.n_iso_bases]
                
                # --- CRITICAL FIX: ENTROPY REGULARIZATION ---
                # We recover the probability distribution of the spectrum
                # irrespective of the fiber fraction magnitude.
                iso_sum = torch.sum(iso_weights, dim=1, keepdim=True) + 1e-9
                iso_prob = iso_weights / iso_sum
                
                # Entropy = - sum(p * log(p))
                # Minimizing entropy -> makes spectrum "peaky" (sparse)
                # without forcing the total sum to zero.
                loss_entropy = -torch.sum(iso_prob * torch.log(iso_prob + 1e-9), dim=1).mean()
                
                # Reconstruction
                signal_recon = decoder(params_pred)
                loss_recon = recon_loss_fn(signal_recon, clean_signal)
                
                # Total Loss: 
                # Entropy weight 0.005 keeps spectrum clean but allows Hindered
                total_loss = loss_recon + (0.005 * loss_entropy)
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += loss_recon.item()
            
            pbar.set_postfix({"Recon": f"{epoch_loss/len(dataloader):.6f}"})
        
        # 4. Inference
        print("[DeepSolver] Inference...")
        encoder.eval()
        full_loader = DataLoader(dataset, batch_size=4096, shuffle=False)
        all_preds = []
        with torch.no_grad():
            for batch in full_loader:
                all_preds.append(encoder(batch[0].to(DEVICE)).cpu().numpy())
        
        flat_results = np.concatenate(all_preds, axis=0)
        return self._pack_results(flat_results, X_vol, Y_vol, Z_vol, mask)

    def _pack_results(self, flat_results, X, Y, Z, mask):
        # Helper to create 3D volumes
        def to_3d(flat_arr):
            vol = np.zeros((X, Y, Z), dtype=np.float32)
            vol[mask] = flat_arr
            return vol
            
        n = self.n_iso_bases
        iso_w = flat_results[:, :n]
        f_fib = flat_results[:, n]
        d_ax  = flat_results[:, n+3]
        d_rad = flat_results[:, n+4]
        
        # Spectral Aggregation
        grid = np.linspace(0, 3.0e-3, n)
        mask_res = grid <= 0.3e-3
        mask_hin = (grid > 0.3e-3) & (grid <= 2.0e-3)
        mask_wat = grid > 2.0e-3
        
        return {
            'restricted_fraction': to_3d(np.sum(iso_w[:, mask_res], axis=1)),
            'hindered_fraction':   to_3d(np.sum(iso_w[:, mask_hin], axis=1)),
            'water_fraction':      to_3d(np.sum(iso_w[:, mask_wat], axis=1)),
            'fiber_fraction':      to_3d(f_fib),
            'axial_diffusivity':   to_3d(d_ax),
            'radial_diffusivity':  to_3d(d_rad),
        }