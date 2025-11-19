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
        
        # Fixed buffers
        self.register_buffer('bvals', torch.tensor(bvals, dtype=torch.float32))
        self.register_buffer('bvecs', torch.tensor(bvecs, dtype=torch.float32))
        
        # Fixed spectral grid [0, 3.0] um^2/ms
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

        # 1. Anisotropic Component
        fiber_dir = torch.stack([
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta)
        ], dim=1)

        cos_angle = torch.matmul(fiber_dir, self.bvecs.T)
        d_app_fiber = d_rad.unsqueeze(1) + (d_ax.unsqueeze(1) - d_rad.unsqueeze(1)) * (cos_angle ** 2)
        
        signal_fiber = f_fiber.unsqueeze(1) * torch.exp(-self.bvals.unsqueeze(0) * d_app_fiber)

        # 2. Isotropic Component
        basis_iso = torch.exp(-torch.ger(self.bvals, self.iso_diffusivities))
        signal_iso = torch.matmul(f_iso_weights, basis_iso.T)

        return signal_fiber + signal_iso


class DBSI_RegularizedMLP(nn.Module):
    """
    MLP Encoder with Hierarchical Fraction Estimation.
    Decouples Fiber fraction prediction from Isotropic distribution to prevent
    dilution of the fiber signal in the early training stages.
    """
    def __init__(self, n_input_meas: int, n_iso_bases: int = 50, dropout_rate: float = 0.1):
        super().__init__()
        self.n_iso = n_iso_bases
        self.n_output = n_iso_bases + 5 
        
        self.net = nn.Sequential(
            nn.Linear(n_input_meas, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, self.n_output)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        raw = self.net(x)
        
        # --- 1. Hierarchical Fractions (The Fix) ---
        # Instead of one big Softmax, we split the decision:
        # A) How much is Fiber? (Sigmoid -> starts at 0.5 probability)
        f_fiber_prob = self.sigmoid(raw[:, self.n_iso])
        
        # B) How is the REST distributed among isotropic bases? (Softmax)
        # We allocate the remaining probability (1 - f_fiber) to the isotropic spectrum
        f_iso_total_prob = 1.0 - f_fiber_prob
        iso_logits = raw[:, :self.n_iso]
        iso_distribution = torch.softmax(iso_logits, dim=1)
        
        # Combine: f_iso_i = (1 - f_fiber) * distribution_i
        f_iso = f_iso_total_prob.unsqueeze(1) * iso_distribution
        
        # Ensure flat fiber fraction for output concatenation
        f_fiber = f_fiber_prob
        
        # --- 2. Angles ---
        theta = self.sigmoid(raw[:, self.n_iso + 1]) * np.pi        
        phi   = (self.sigmoid(raw[:, self.n_iso + 2]) - 0.5) * 2 * np.pi
        
        # --- 3. Diffusivities (Delta-Parametrization from previous step) ---
        # We keep this because it works well for the means.
        d_rad_raw = self.sigmoid(raw[:, self.n_iso + 4])
        d_rad = d_rad_raw * 1.5e-3 

        delta_raw = self.sigmoid(raw[:, self.n_iso + 3])
        d_delta = (delta_raw * 2.4e-3) + 0.1e-3 # Gap 0.1 to force anisotropy
        
        d_ax = d_rad + d_delta

        return torch.cat([
            f_iso, 
            f_fiber.unsqueeze(1), 
            theta.unsqueeze(1), 
            phi.unsqueeze(1), 
            d_ax.unsqueeze(1), 
            d_rad.unsqueeze(1)
        ], dim=1)

class DBSI_DeepSolver:
    """
    Self-Supervised Solver with Denoising Regularization.
    """
    def __init__(self, 
                 n_iso_bases: int = 20,
                 epochs: int = 150,      
                 batch_size: int = 2048, 
                 learning_rate: float = 1e-3,
                 noise_injection_level: float = 0.02):
        
        self.n_iso_bases = n_iso_bases
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.noise_level = noise_injection_level
        
    def fit_volume(self, volume: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
        
        print(f"[DeepSolver] Starting training on device: {DEVICE}")
        print(f"[DeepSolver] Geometry: Delta-Parameterization (No Dead Gradients)")
        
        # 1. Data Preparation
        X_vol, Y_vol, Z_vol, N_meas = volume.shape
        valid_signals = volume[mask]
        
        # Robust Normalization
        valid_signals = np.nan_to_num(valid_signals, nan=0.0, posinf=0.0, neginf=0.0)
        valid_signals = np.maximum(valid_signals, 0.0)
        
        b0_idx = np.where(bvals < 50)[0]
        if len(b0_idx) > 0:
            s0 = np.mean(valid_signals[:, b0_idx], axis=1, keepdims=True)
            s0[s0 < 1e-3] = 1.0
            valid_signals = valid_signals / s0
            valid_signals = np.clip(valid_signals, 0, 1.5)
        
        dataset = TensorDataset(torch.tensor(valid_signals, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 2. Model Initialization
        encoder = DBSI_RegularizedMLP(
            n_input_meas=N_meas, 
            n_iso_bases=self.n_iso_bases,
            dropout_rate=0.1
        ).to(DEVICE)
        
        decoder = DBSI_PhysicsDecoder(
            bvals, 
            bvecs, 
            n_iso_bases=self.n_iso_bases
        ).to(DEVICE)
        
        optimizer = optim.AdamW(encoder.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.L1Loss()
        
        # 3. Training Loop
        print(f"[DeepSolver] Training on {len(valid_signals)} voxels...")
        encoder.train()
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False, file=sys.stdout):
                clean_signal = batch[0].to(DEVICE)
                
                noise = torch.randn_like(clean_signal) * self.noise_level
                noisy_input = clean_signal + noise
                
                optimizer.zero_grad()
                
                params_pred = encoder(noisy_input)
                signal_recon = decoder(params_pred)
                
                loss = loss_fn(signal_recon, clean_signal)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: Avg Loss = {epoch_loss / len(dataloader):.6f}")
        
        # 4. Inference
        print("[DeepSolver] Generating volumetric maps...")
        encoder.eval()
        full_loader = DataLoader(dataset, batch_size=self.batch_size*2, shuffle=False)
        all_preds = []
        
        with torch.no_grad():
            for batch in full_loader:
                x = batch[0].to(DEVICE)
                preds = encoder(x)
                all_preds.append(preds.cpu().numpy())
                
        flat_results = np.concatenate(all_preds, axis=0)
        
        # 5. Reconstruction
        def to_3d(flat_arr):
            vol = np.zeros((X_vol, Y_vol, Z_vol), dtype=np.float32)
            vol[mask] = flat_arr
            return vol
            
        iso_weights = flat_results[:, :self.n_iso_bases]
        fiber_frac  = flat_results[:, self.n_iso_bases]
        theta       = flat_results[:, self.n_iso_bases+1]
        phi         = flat_results[:, self.n_iso_bases+2]
        d_ax        = flat_results[:, self.n_iso_bases+3]
        d_rad       = flat_results[:, self.n_iso_bases+4]
        
        # Spectral Aggregation
        grid = np.linspace(0, 3.0e-3, self.n_iso_bases)
        mask_res = grid <= 0.3e-3
        mask_hin = (grid > 0.3e-3) & (grid <= 2.0e-3)
        mask_wat = grid > 2.0e-3
        
        f_res = np.sum(iso_weights[:, mask_res], axis=1)
        f_hin = np.sum(iso_weights[:, mask_hin], axis=1)
        f_wat = np.sum(iso_weights[:, mask_wat], axis=1)
        
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