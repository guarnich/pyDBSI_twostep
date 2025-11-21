# dbsi_toolbox/deep_learning.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import sys
from typing import Dict, Tuple

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. GENERATORE SINTETICO
# ==========================================
class SyntheticGenerator:
    """Generates synthetic DBSI signals for supervised pre-training."""
    def __init__(self, bvals: np.ndarray, bvecs: np.ndarray, n_iso_bases: int = 20):
        self.bvals = bvals
        self.bvecs = bvecs
        self.n_meas = len(bvals)
        self.iso_grid = np.linspace(0, 3.0e-3, n_iso_bases)
        self.n_iso = n_iso_bases

    def generate_batch(self, batch_size: int, snr: float = 30.0) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Sample Random Parameters
        f_fiber = np.random.uniform(0.0, 0.8, batch_size)
        
        # Remaining fraction for isotropic
        f_iso_total = 1.0 - f_fiber
        
        # Split isotropic into Restricted, Hindered, Water
        # Weights: [Restricted, Hindered, Water]
        iso_ratios = np.random.dirichlet((1, 2, 1), batch_size) 
        f_res = f_iso_total * iso_ratios[:, 0]
        f_hin = f_iso_total * iso_ratios[:, 1]
        f_wat = f_iso_total * iso_ratios[:, 2]

        # Fiber Orientation (Random on Sphere)
        theta = np.arccos(2 * np.random.rand(batch_size) - 1)
        phi = 2 * np.pi * np.random.rand(batch_size)
        fiber_dir = np.stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ], axis=1)

        # Diffusivities
        d_ax = np.random.uniform(1.5e-3, 2.2e-3, batch_size)
        d_rad = np.random.uniform(0.1e-3, 0.6e-3, batch_size)
        
        # Isotropic Diffusivities (Simplified for simulation)
        d_res = 0.1e-3
        d_hin = 1.0e-3
        d_wat = 3.0e-3

        # 2. Generate Signal (Forward Model)
        signals = np.zeros((batch_size, self.n_meas))
        
        # Precompute angles
        cos_angles = np.dot(fiber_dir, self.bvecs.T) # (Batch, N_meas)
        d_app_fiber = d_rad[:, None] + (d_ax[:, None] - d_rad[:, None]) * (cos_angles**2)
        
        sig_fib = np.exp(-self.bvals[None, :] * d_app_fiber)
        sig_res = np.exp(-self.bvals[None, :] * d_res)
        sig_hin = np.exp(-self.bvals[None, :] * d_hin)
        sig_wat = np.exp(-self.bvals[None, :] * d_wat)
        
        clean_signal = (
            f_fiber[:, None] * sig_fib +
            f_res[:, None] * sig_res +
            f_hin[:, None] * sig_hin +
            f_wat[:, None] * sig_wat
        )

        # 3. Add Rician Noise
        sigma = 1.0 / snr
        noise_real = np.random.normal(0, sigma, clean_signal.shape)
        noise_imag = np.random.normal(0, sigma, clean_signal.shape)
        noisy_signal = np.sqrt((clean_signal + noise_real)**2 + noise_imag**2)
        
        # 4. Prepare Labels (We need to match the network output format roughly)
        # Labels: [f_restricted, f_hindered, f_water, f_fiber]
        labels = np.stack([f_res, f_hin, f_wat, f_fiber], axis=1)

        return torch.tensor(noisy_signal, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

# ==========================================
# 2. DECODER FISICO (Invariato)
# ==========================================
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

# ==========================================
# 3. RETE NEURALE (Migliorata)
# ==========================================
class DBSI_RegularizedMLP(nn.Module):
    def __init__(self, n_input_meas: int, n_iso_bases: int = 20, dropout_rate: float = 0.05):
        super().__init__()
        self.n_iso = n_iso_bases
        
        # Defines masks for aggregating micro-fractions into macro-fractions
        grid = np.linspace(0, 3.0e-3, n_iso_bases)
        self.idx_res = torch.tensor(np.where(grid <= 0.3e-3)[0], dtype=torch.long)
        self.idx_hin = torch.tensor(np.where((grid > 0.3e-3) & (grid <= 2.0e-3))[0], dtype=torch.long)
        self.idx_wat = torch.tensor(np.where(grid > 2.0e-3)[0], dtype=torch.long)
        
        self.backbone = nn.Sequential(
            nn.Linear(n_input_meas, 512), # Increased Width
            nn.LayerNorm(512), nn.ELU(), nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256), nn.ELU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.LayerNorm(128), nn.ELU()
        )
        
        # Output Heads
        self.head_fractions = nn.Linear(128, 4) # Res, Hin, Wat, Fib
        self.head_iso_micro = nn.Linear(128, n_iso_bases)
        self.head_geom = nn.Linear(128, 4) # Theta, Phi, Dax, Drad
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        feat = self.backbone(x)
        
        # Macro Fractions (Sum to 1)
        probs = self.softmax(self.head_fractions(feat))
        f_res_macro = probs[:, 0:1]
        f_hin_macro = probs[:, 1:2]
        f_wat_macro = probs[:, 2:3]
        f_fiber     = probs[:, 3]
        
        # Micro Isotropic Distributions (Conditional Softmax)
        micro_logits = self.head_iso_micro(feat)
        
        # Create sparse-like distributions inside each compartment
        res_dist = self.softmax(micro_logits[:, self.idx_res])
        w_res = res_dist * f_res_macro
        
        hin_dist = self.softmax(micro_logits[:, self.idx_hin])
        w_hin = hin_dist * f_hin_macro
        
        wat_dist = self.softmax(micro_logits[:, self.idx_wat])
        w_wat = wat_dist * f_wat_macro
        
        f_iso_weights = torch.cat([w_res, w_hin, w_wat], dim=1)
        
        # Geometry
        geom = self.head_geom(feat)
        theta = self.sigmoid(geom[:, 0]) * np.pi
        phi   = (self.sigmoid(geom[:, 1]) - 0.5) * 2 * np.pi
        
        # Relaxed Constraints for Diffusivity
        d_ax  = self.sigmoid(geom[:, 2]) * 2.5e-3 + 0.5e-3  # [0.5, 3.0]
        d_rad = self.sigmoid(geom[:, 3]) * 1.5e-3           # [0.0, 1.5]
        d_ax  = torch.max(d_ax, d_rad + 1e-7)               # Ensure D_ax >= D_rad

        return torch.cat([
            f_iso_weights, 
            f_fiber.unsqueeze(1), 
            theta.unsqueeze(1), 
            phi.unsqueeze(1), 
            d_ax.unsqueeze(1), 
            d_rad.unsqueeze(1)
        ], dim=1)

    def get_macro_fractions(self, model_output):
        """Helper to extract summed macro fractions from the full output vector."""
        n = self.n_iso
        iso_weights = model_output[:, :n]
        f_fib = model_output[:, n]
        
        f_res = torch.sum(iso_weights[:, self.idx_res], dim=1)
        f_hin = torch.sum(iso_weights[:, self.idx_hin], dim=1)
        f_wat = torch.sum(iso_weights[:, self.idx_wat], dim=1)
        
        return torch.stack([f_res, f_hin, f_wat, f_fib], dim=1)

# ==========================================
# 4. SOLVER CON PRE-TRAINING
# ==========================================
class DBSI_DeepSolver:
    def __init__(self, n_iso_bases: int = 20, epochs: int = 100, batch_size: int = 4096, 
                 learning_rate: float = 1e-3, noise_injection_level: float = 0.03,
                 pretrain_steps: int = 2000):
        self.n_iso_bases = n_iso_bases
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.noise_level = noise_injection_level
        self.pretrain_steps = pretrain_steps
        
    def fit_volume(self, volume: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
        print(f"[DeepSolver] Strategy: Synthetic Pre-training -> Self-Supervised Fine-Tuning")
        
        # --- DATA PREP ---
        X_vol, Y_vol, Z_vol, N_meas = volume.shape
        valid_signals = volume[mask]
        
        # Normalize Signal
        b0_idx = np.where(bvals < 50)[0]
        if len(b0_idx) > 0:
            s0 = np.mean(valid_signals[:, b0_idx], axis=1, keepdims=True)
            s0[s0 < 1e-4] = 1.0 
            valid_signals = valid_signals / s0
            valid_signals = np.clip(valid_signals, 0, 1.5)
        
        dataset = TensorDataset(torch.tensor(valid_signals, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        # --- MODEL SETUP ---
        encoder = DBSI_RegularizedMLP(N_meas, self.n_iso_bases).to(DEVICE)
        decoder = DBSI_PhysicsDecoder(bvals, bvecs, self.n_iso_bases).to(DEVICE)
        synth_gen = SyntheticGenerator(bvals, bvecs, self.n_iso_bases)
        
        optimizer = optim.AdamW(encoder.parameters(), lr=self.lr)
        loss_mse = nn.MSELoss()
        loss_l1 = nn.L1Loss()

        # ============================
        # PHASE 1: SUPERVISED PRE-TRAINING
        # ============================
        print(f"Starting Synthetic Pre-training ({self.pretrain_steps} steps)...")
        encoder.train()
        pbar_pre = tqdm(range(self.pretrain_steps), desc="Pre-training", unit="step", file=sys.stdout)
        
        for step in pbar_pre:
            # Generate Synthetic Batch
            sim_sig, sim_labels = synth_gen.generate_batch(self.batch_size, snr=30.0)
            sim_sig = sim_sig.to(DEVICE)
            sim_labels = sim_labels.to(DEVICE) # [Res, Hin, Wat, Fib]
            
            optimizer.zero_grad()
            
            # Forward
            preds = encoder(sim_sig)
            
            # Extract predicted macro fractions
            pred_fractions = encoder.get_macro_fractions(preds)
            
            # Supervised Loss: Force network to output correct fractions
            l_supervised = loss_mse(pred_fractions, sim_labels)
            
            # Consistency Loss: Physics check
            recon_sig = decoder(preds)
            l_recon = loss_mse(recon_sig, sim_sig)
            
            loss = l_supervised + 0.1 * l_recon
            
            loss.backward()
            optimizer.step()
            
            if step % 100 == 0:
                pbar_pre.set_postfix({"SupLoss": f"{l_supervised.item():.4f}"})

        # ============================
        # PHASE 2: SELF-SUPERVISED FINE-TUNING
        # ============================
        print("\nStarting Fine-Tuning on Real Data...")
        current_noise = self.noise_level
        pbar_ft = tqdm(range(self.epochs), desc="Fine-Tuning", unit="epoch", file=sys.stdout)
        
        for epoch in pbar_ft:
            batch_loss = 0.0
            
            # Decay noise
            if epoch > self.epochs // 2: current_noise = self.noise_level * 0.5
            if epoch > int(self.epochs * 0.8): current_noise = 0.0
            
            for batch in dataloader:
                clean = batch[0].to(DEVICE)
                noise = torch.randn_like(clean) * current_noise
                
                optimizer.zero_grad()
                preds = encoder(clean + noise)
                
                # Reconstruction Loss
                recon = decoder(preds)
                l_fit = loss_mse(recon, clean)
                
                # Sparsity Regularization on Isotropic Weights (L1)
                # This helps push small isotropic components to zero (mimicking NNLS)
                iso_weights = preds[:, :self.n_iso_bases]
                l_sparsity = loss_l1(iso_weights, torch.zeros_like(iso_weights))
                
                loss = l_fit + (0.001 * l_sparsity)
                
                loss.backward()
                optimizer.step()
                batch_loss += l_fit.item()
            
            pbar_ft.set_postfix({"ReconMSE": f"{batch_loss/len(dataloader):.6f}"})
            
        # --- INFERENCE ---
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