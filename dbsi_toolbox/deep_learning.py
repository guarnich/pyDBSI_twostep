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
# 1. GENERATORE SINTETICO (Protocol-Specific)
# ==========================================
class DBSISyntheticGenerator:
    """
    Genera segnali DBSI sintetici basati ESATTAMENTE sui bvals/bvecs 
    dell'immagine in input. Questo garantisce la validità fisica.
    """
    def __init__(self, bvals: np.ndarray, bvecs: np.ndarray, n_iso_bases: int = 20):
        self.bvals = bvals
        self.bvecs = bvecs
        self.n_meas = len(bvals)
        self.n_iso = n_iso_bases

    def generate_batch(self, batch_size: int, snr: float = 30.0) -> Tuple[torch.Tensor, torch.Tensor]:
        # Logica "Scenario-Based" per coprire tutti i casi clinici (Sano, Edema, Infiammazione)
        n1 = batch_size // 3
        n2 = batch_size // 3
        n3 = batch_size - n1 - n2
        
        # 1. Fibra Dominante
        f_fib_1 = np.random.uniform(0.4, 0.9, n1)
        iso_1 = np.random.dirichlet((1, 1, 1), n1)
        
        # 2. Hindered Dominante (Edema)
        f_fib_2 = np.random.uniform(0.0, 0.3, n2)
        iso_2 = np.random.dirichlet((1, 8, 1), n2) 
        
        # 3. Restricted Dominante (Infiammazione)
        f_fib_3 = np.random.uniform(0.1, 0.5, n3)
        iso_3 = np.random.dirichlet((8, 1, 1), n3)

        f_fiber = np.concatenate([f_fib_1, f_fib_2, f_fib_3])
        iso_ratios = np.concatenate([iso_1, iso_2, iso_3])
        
        # Shuffle
        perm = np.random.permutation(batch_size)
        f_fiber = f_fiber[perm]
        iso_ratios = iso_ratios[perm]

        # Calcolo Frazioni Reali
        f_iso_total = 1.0 - f_fiber
        f_res = f_iso_total * iso_ratios[:, 0]
        f_hin = f_iso_total * iso_ratios[:, 1]
        f_wat = f_iso_total * iso_ratios[:, 2]

        # Parametri Fisici
        theta = np.arccos(2 * np.random.rand(batch_size) - 1)
        phi = 2 * np.pi * np.random.rand(batch_size)
        fiber_dir = np.stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ], axis=1)

        d_ax = np.random.uniform(1.5e-3, 2.2e-3, batch_size)
        d_rad = np.random.uniform(0.1e-3, 0.6e-3, batch_size)
        d_res = 0.1e-3 # Restricted fisso basso
        d_hin = 1.5e-3 # Hindered medio
        d_wat = 3.0e-3 # Acqua libera

        # Forward Model: Usiamo i bvecs/bvals REALI
        signals = np.zeros((batch_size, self.n_meas))
        cos_angles = np.dot(fiber_dir, self.bvecs.T)
        
        # Anisotropo
        d_app_fiber = d_rad[:, None] + (d_ax[:, None] - d_rad[:, None]) * (cos_angles**2)
        sig_fib = np.exp(-self.bvals[None, :] * d_app_fiber)
        
        # Isotropo
        sig_res = np.exp(-self.bvals[None, :] * d_res)
        sig_hin = np.exp(-self.bvals[None, :] * d_hin)
        sig_wat = np.exp(-self.bvals[None, :] * d_wat)
        
        clean_signal = (
            f_fiber[:, None] * sig_fib +
            f_res[:, None] * sig_res +
            f_hin[:, None] * sig_hin +
            f_wat[:, None] * sig_wat
        )

        # Rumore Rician
        sigma = 1.0 / snr
        noise_real = np.random.normal(0, sigma, clean_signal.shape)
        noise_imag = np.random.normal(0, sigma, clean_signal.shape)
        noisy_signal = np.sqrt((clean_signal + noise_real)**2 + noise_imag**2)
        
        # Normalizzazione (Cruciale per DL)
        b0_mean = np.mean(noisy_signal[:, self.bvals < 50], axis=1, keepdims=True) + 1e-9
        noisy_signal = noisy_signal / b0_mean

        # Target: [Restricted, Hindered, Water, Fiber]
        labels = np.stack([f_res, f_hin, f_wat, f_fiber], axis=1)

        return torch.tensor(noisy_signal, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

# ==========================================
# 2. RETE NEURALE SEMPLIFICATA (Solo Priors)
# ==========================================
class DBSI_PriorNet(nn.Module):
    def __init__(self, n_input_meas: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input_meas, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            
            nn.Linear(128, 4), # Output: [f_res, f_hin, f_wat, f_fib]
            nn.Softmax(dim=1)  # Somma a 1
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. REGOLARIZZATORE (Gestore)
# ==========================================
class DBSI_DeepRegularizer:
    """
    Addestra una rete neurale su dati sintetici generati ad-hoc per il protocollo corrente.
    Fornisce i 'priors' per il fit NLLS.
    """
    def __init__(self, bvals: np.ndarray, bvecs: np.ndarray, epochs: int = 50, 
                 batch_size: int = 1024, lr: float = 1e-3):
        self.bvals = bvals
        self.bvecs = bvecs
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = None
        self.generator = DBSISyntheticGenerator(bvals, bvecs)

    def train_on_synthetic(self, n_samples=50000):
        print(f"[DL-Reg] Generating {n_samples} synthetic samples (Protocol-Specific)...")
        X, Y = self.generator.generate_batch(n_samples, snr=30.0)
        
        dataset = TensorDataset(X, Y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = DBSI_PriorNet(n_input_meas=len(self.bvals)).to(DEVICE)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss() # MSE sulle frazioni
        
        self.model.train()
        print(f"[DL-Reg] Training Prior Network on {DEVICE}...")
        
        # Training loop rapido
        for _ in tqdm(range(self.epochs), desc="Training Priors", file=sys.stdout):
            for bx, by in loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                preds = self.model(bx)
                loss = criterion(preds, by)
                loss.backward()
                optimizer.step()

    def predict_volume(self, volume: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
        if self.model is None: raise RuntimeError("Model not trained.")
        
        self.model.eval()
        X_dim, Y_dim, Z_dim, N_meas = volume.shape
        
        # Estrai voxel validi
        valid_signals = volume[mask]
        
        # Normalizzazione S0
        b0_idx = self.bvals < 50
        if np.sum(b0_idx) > 0:
            s0 = np.mean(valid_signals[:, b0_idx], axis=1, keepdims=True)
            valid_signals = valid_signals / (s0 + 1e-9)
            
        # Inferenza
        dataset = TensorDataset(torch.tensor(valid_signals, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=4096, shuffle=False)
        
        preds_list = []
        with torch.no_grad():
            for batch in loader:
                bx = batch[0].to(DEVICE)
                preds_list.append(self.model(bx).cpu().numpy())
                
        flat_preds = np.concatenate(preds_list, axis=0)
        
        # Ricostruzione Mappe 3D
        maps = {}
        keys = ['dl_restricted_fraction', 'dl_hindered_fraction', 'dl_water_fraction', 'dl_fiber_fraction']
        
        for i, k in enumerate(keys):
            vol = np.zeros((X_dim, Y_dim, Z_dim))
            vol[mask] = flat_preds[:, i]
            maps[k] = vol
            
        return maps