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
    Implements the exact DBSI equation (Wang et al., 2011).
    """
    def __init__(self, bvals: np.ndarray, bvecs: np.ndarray, n_iso_bases: int = 50):
        super().__init__()
        
        # Fixed buffers for acquisition geometry
        self.register_buffer('bvals', torch.tensor(bvals, dtype=torch.float32))
        self.register_buffer('bvecs', torch.tensor(bvecs, dtype=torch.float32))
        
        # Fixed spectral grid [0, 3.0] um^2/ms
        # Using slightly wider range or log-spacing can sometimes help, 
        # but linear is standard for DBSI.
        iso_grid = torch.linspace(0, 3.0e-3, n_iso_bases)
        self.register_buffer('iso_diffusivities', iso_grid)
        
        self.n_iso = n_iso_bases

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        # params structure:
        # [0..N-1]: f_iso (N bases)
        # [N]:      f_fiber
        # [N+1]:    theta
        # [N+2]:    phi
        # [N+3]:    d_ax
        # [N+4]:    d_rad

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
        
        # Fiber signal: f_fib * exp(-b * D_app)
        signal_fiber = f_fiber.unsqueeze(1) * torch.exp(-self.bvals.unsqueeze(0) * d_app_fiber)

        # 2. Isotropic Component (Spectrum)
        basis_iso = torch.exp(-torch.ger(self.bvals, self.iso_diffusivities))
        signal_iso = torch.matmul(f_iso_weights, basis_iso.T)

        # 3. Total Signal
        return signal_fiber + signal_iso


class DBSI_RegularizedMLP(nn.Module):
    """
    Updated MLP Encoder with "Split Fraction" Architecture.
    Decouples Fiber Fraction estimation from Isotropic Spectrum distribution
    to prevent fiber underestimation.
    """
    def __init__(self, n_input_meas: int, n_iso_bases: int = 50, dropout_rate: float = 0.1):
        super().__init__()
        self.n_iso = n_iso_bases
        
        # Outputs:
        # 1 for Fiber Fraction (Sigmoid)
        # n_iso_bases for Isotropic Distribution (Softmax)
        # 2 for Angles
        # 2 for Diffusivities
        # FIX: 'n_iso' was typo, changed to 'n_iso_bases'
        self.n_output_features = 1 + n_iso_bases + 2 + 2 
        
        # Deeper/Wider network to capture subtle features
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
        # Softmax for the isotropic spectrum only
        self.softmax_iso = nn.Softmax(dim=1) 

    def forward(self, x):
        raw = self.net(x)
        
        # --- SPLIT FRACTION ESTIMATION ---
        # Index 0: Fiber Fraction Logit
        # Index 1..N: Isotropic Spectrum Logits
        
        # 1. Calculate Fiber Fraction (0 to 1)
        f_fiber_raw = raw[:, 0]
        f_fiber = self.sigmoid(f_fiber_raw)
        
        # 2. Calculate Isotropic Distribution (Sums to 1)
        # FIX: Use self.n_iso correctly for slicing
        iso_logits = raw[:, 1:self.n_iso + 1]
        iso_dist = self.softmax_iso(iso_logits)
        
        # 3. Scale Isotropic Fractions
        # The remaining signal (1 - f_fiber) is distributed among isotropic bases
        f_iso_total = 1.0 - f_fiber
        f_iso_weights = iso_dist * f_iso_total.unsqueeze(1)
        
        # --- Geometric Parameters ---
        # Indices start after the fractions
        idx_geom = self.n_iso + 1
        
        # Angles
        theta = self.sigmoid(raw[:, idx_geom]) * np.pi        # [0, pi]
        phi   = (self.sigmoid(raw[:, idx_geom + 1]) - 0.5) * 2 * np.pi # [-pi, pi]
        
        # Diffusivities
        # D_ax: [0.5, 3.0] - Preventing collapse to 0 helps separate from restricted iso
        d_ax = self.sigmoid(raw[:, idx_geom + 2]) * 2.5e-3 + 0.5e-3 
        
        # D_rad: [0.0, 1.5] - Standard range
        d_rad = self.sigmoid(raw[:, idx_geom + 3]) * 1.5e-3
        
        # Constraint: D_ax >= D_rad
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
    """
    Self-Supervised Solver with Adaptive Noise and Sparsity Regularization.
    """
    def __init__(self, 
                 n_iso_bases: int = 50, 
                 epochs: int = 200,  # Increased epochs for better convergence
                 batch_size: int = 1024, 
                 learning_rate: float = 1e-3,
                 noise_injection_level: float = 0.03): # Increased noise slightly for robustness
        
        self.n_iso_bases = n_iso_bases
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.noise_level = noise_injection_level
        
    def fit_volume(self, volume: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
        
        print(f"[DeepSolver] Starting self-supervised training on device: {DEVICE}")
        print(f"[DeepSolver] Strategy: Split Fractions + L1 Sparsity")
        
        # 1. Data Preparation
        X_vol, Y_vol, Z_vol, N_meas = volume.shape
        valid_signals = volume[mask]
        
        # Robust S0 normalization
        b0_idx = np.where(bvals < 50)[0]
        if len(b0_idx) > 0:
            s0 = np.mean(valid_signals[:, b0_idx], axis=1, keepdims=True)
            s0[s0 < 1e-4] = 1.0 
            valid_signals = valid_signals / s0
            # Clip outliers
            valid_signals = np.clip(valid_signals, 0, 1.5)
        
        dataset = TensorDataset(torch.tensor(valid_signals, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        # 2. Model Initialization
        encoder = DBSI_RegularizedMLP(
            n_input_meas=N_meas, 
            n_iso_bases=self.n_iso_bases,
            dropout_rate=0.05 # Reduced dropout to reduce variance in output
        ).to(DEVICE)
        
        decoder = DBSI_PhysicsDecoder(
            bvals, 
            bvecs, 
            n_iso_bases=self.n_iso_bases
        ).to(DEVICE)
        
        optimizer = optim.AdamW(encoder.parameters(), lr=self.lr, weight_decay=1e-5)
        
        # Loss Functions
        recon_loss_fn = nn.MSELoss() # MSE encourages fitting the mean (less noise)
        
        # 3. Training Loop
        encoder.train()
        
        # Adaptive Noise Schedule: Start high, reduce to refine
        current_noise = self.noise_level
        
        for epoch in range(self.epochs):
            epoch_loss_recon = 0.0
            epoch_loss_sparse = 0.0
            
            # Decay noise halfway through
            if epoch == self.epochs // 2:
                current_noise = self.noise_level * 0.5
            
            for batch in dataloader: # tqdm removed for cleaner logs, add back if needed
                clean_signal = batch[0].to(DEVICE)
                
                # Noise Injection (Denoising Autoencoder)
                noise = torch.randn_like(clean_signal) * current_noise
                noisy_input = clean_signal + noise
                
                optimizer.zero_grad()
                
                # A. Predict Parameters
                params_pred = encoder(noisy_input)
                
                # Extract fractions for sparsity regularization
                # Only penalize Isotropic components to encourage Fiber usage?
                # Or penalize everything to encourage simplicity?
                # Let's calculate entropy of the isotropic distribution to encourage sharpness
                # isotropic weights are params_pred[:, :self.n_iso_bases]
                
                iso_weights = params_pred[:, :self.n_iso_bases]
                
                # B. Reconstruct Signal
                signal_recon = decoder(params_pred)
                
                # C. Losses
                # 1. Data Fidelity
                loss_recon = recon_loss_fn(signal_recon, clean_signal)
                
                # 2. Sparsity on Isotropic Spectrum (L1)
                # Helps reducing "smearing" of the spectrum, reducing zeros in fiber
                # by forcing isotropic parts to be efficient.
                loss_sparsity = torch.mean(torch.sum(torch.abs(iso_weights), dim=1))
                
                # Total Loss
                # Weighting sparsity is tricky. Too high -> underfitting. 
                # 0.001 is conservative.
                total_loss = loss_recon + (0.001 * loss_sparsity)
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss_recon += loss_recon.item()
                epoch_loss_sparse += loss_sparsity.item()
            
            if (epoch + 1) % 20 == 0:
                avg_recon = epoch_loss_recon / len(dataloader)
                print(f"[Epoch {epoch+1}/{self.epochs}] Recon Loss: {avg_recon:.6f}")
        
        # 4. Inference
        print("[DeepSolver] Inference on full volume...")
        encoder.eval()
        # Use larger batch for inference
        full_loader = DataLoader(dataset, batch_size=4096, shuffle=False)
        all_preds = []
        
        with torch.no_grad():
            for batch in full_loader:
                x = batch[0].to(DEVICE)
                # No noise during inference
                preds = encoder(x)
                all_preds.append(preds.cpu().numpy())
                
        flat_results = np.concatenate(all_preds, axis=0)
        
        # 5. Reconstruct Maps
        return self._pack_results(flat_results, X_vol, Y_vol, Z_vol, mask)

    def _pack_results(self, flat_results, X, Y, Z, mask):
        n_iso = self.n_iso_bases
        
        def to_3d(flat_arr):
            vol = np.zeros((X, Y, Z), dtype=np.float32)
            vol[mask] = flat_arr
            return vol
            
        iso_weights = flat_results[:, :n_iso]
        fiber_frac  = flat_results[:, n_iso]
        theta       = flat_results[:, n_iso+1]
        phi         = flat_results[:, n_iso+2]
        d_ax        = flat_results[:, n_iso+3]
        d_rad       = flat_results[:, n_iso+4]
        
        # Spectral Aggregation
        grid = np.linspace(0, 3.0e-3, n_iso)
        mask_res = grid <= 0.3e-3
        mask_hin = (grid > 0.3e-3) & (grid <= 2.0e-3)
        mask_wat = grid > 2.0e-3
        
        f_res = np.sum(iso_weights[:, mask_res], axis=1)
        f_hin = np.sum(iso_weights[:, mask_hin], axis=1)
        f_wat = np.sum(iso_weights[:, mask_wat], axis=1)
        
        # Direction
        dir_x = np.sin(theta) * np.cos(phi)
        dir_y = np.sin(theta) * np.sin(phi)
        dir_z = np.cos(theta)
        
        return {
            'restricted_fraction': to_3d(f_res),
            'hindered_fraction':   to_3d(f_hin),
            'water_fraction':      to_3d(f_wat),
            'fiber_fraction':      to_3d(fiber_frac),
            'axial_diffusivity':   to_3d(d_ax),
            'radial_diffusivity':  to_3d(d_rad),
            'fiber_dir_x':         to_3d(dir_x),
            'fiber_dir_y':         to_3d(dir_y),
            'fiber_dir_z':         to_3d(dir_z),
        }