# dbsi_toolbox/deep_learning.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import sys
from typing import Dict, Tuple

# Rileva GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. GENERATORE SINTETICO "SCENARIO-BASED"
# ==========================================
class DBSISyntheticGenerator:
    """
    Genera dati sintetici bilanciati per coprire i casi clinici reali basandosi
    sul protocollo di acquisizione (bvals/bvecs) specifico.
    """
    def __init__(self, bvals: np.ndarray, bvecs: np.ndarray, d_res_threshold: float = 0.3e-3):
        self.bvals = bvals
        self.bvecs = bvecs
        self.n_meas = len(bvals)
        self.d_res_threshold = d_res_threshold  # Soglia fisica Restricted/Hindered

    def generate_batch(self, batch_size: int, snr: float = 30.0) -> Tuple[torch.Tensor, torch.Tensor]:
        # Dividiamo il batch in 4 scenari clinici distinti
        n_scenarios = 4
        n_per_scen = batch_size // n_scenarios
        
        # --- SCENARIO 1: Fiber Dominant (Sostanza Bianca Sana) ---
        # Fiber > 60%, il resto è rumore isotropo di fondo
        f_fib_1 = np.random.uniform(0.60, 0.95, n_per_scen)
        rem_1 = 1.0 - f_fib_1
        iso_mix_1 = np.random.dirichlet((1, 1, 1), n_per_scen) 
        f_res_1 = rem_1 * iso_mix_1[:, 0]
        f_hin_1 = rem_1 * iso_mix_1[:, 1]
        f_wat_1 = rem_1 * iso_mix_1[:, 2]

        # --- SCENARIO 2: Hindered Dominant (Edema / Danno Tissutale) ---
        # Hindered > 50%
        f_hin_2 = np.random.uniform(0.50, 0.90, n_per_scen)
        rem_2 = 1.0 - f_hin_2
        iso_mix_2 = np.random.dirichlet((1, 2, 1), n_per_scen) # [res, fib, wat]
        f_res_2 = rem_2 * iso_mix_2[:, 0]
        f_fib_2 = rem_2 * iso_mix_2[:, 1]
        f_wat_2 = rem_2 * iso_mix_2[:, 2]

        # --- SCENARIO 3: Restricted Dominant (Alta Cellularità/Infiammazione) ---
        # Restricted > 40% (Scenario critico per DBSI)
        f_res_3 = np.random.uniform(0.40, 0.80, n_per_scen)
        rem_3 = 1.0 - f_res_3
        iso_mix_3 = np.random.dirichlet((1, 1, 1), n_per_scen)
        f_hin_3 = rem_3 * iso_mix_3[:, 0]
        f_fib_3 = rem_3 * iso_mix_3[:, 1]
        f_wat_3 = rem_3 * iso_mix_3[:, 2]

        # --- SCENARIO 4: Water Dominant (CSF / Necrosi) ---
        # Water > 80%
        f_wat_4 = np.random.uniform(0.80, 1.00, n_per_scen)
        rem_4 = 1.0 - f_wat_4
        iso_mix_4 = np.random.dirichlet((1, 1, 1), n_per_scen)
        f_res_4 = rem_4 * iso_mix_4[:, 0]
        f_hin_4 = rem_4 * iso_mix_4[:, 1]
        f_fib_4 = rem_4 * iso_mix_4[:, 2]

        # Concatenazione e Shuffle
        f_fiber = np.concatenate([f_fib_1, f_fib_2, f_fib_3, f_fib_4])
        f_res = np.concatenate([f_res_1, f_res_2, f_res_3, f_res_4])
        f_hin = np.concatenate([f_hin_1, f_hin_2, f_hin_3, f_hin_4])
        f_wat = np.concatenate([f_wat_1, f_wat_2, f_wat_3, f_wat_4])

        perm = np.random.permutation(len(f_fiber))
        f_fiber, f_res, f_hin, f_wat = f_fiber[perm], f_res[perm], f_hin[perm], f_wat[perm]

        # --- Parametri Fisici (Strict Bounds) ---
        batch_real_size = len(f_fiber)
        
        # Restricted: [0, threshold] (Es. 0 - 0.3e-3)
        d_res = np.random.uniform(0.0, self.d_res_threshold, batch_real_size)
        
        # Hindered: [threshold + gap, 2.5] (Es. 0.4 - 2.5e-3)
        # Lasciamo un piccolo gap (0.1e-3) per aiutare la rete a separare le classi
        d_hin = np.random.uniform(self.d_res_threshold + 0.1e-3, 2.5e-3, batch_real_size)
        
        d_wat = 3.0e-3 # Acqua libera a 37°C
        
        # Fiber: [1.0, 2.5] axial, [0.1, 0.6] radial
        d_ax = np.random.uniform(1.0e-3, 2.5e-3, batch_real_size)
        d_rad = np.random.uniform(0.1e-3, 0.6e-3, batch_real_size)

        # Orientamento Fibra Casuale
        theta = np.arccos(2 * np.random.rand(batch_real_size) - 1)
        phi = 2 * np.pi * np.random.rand(batch_real_size)
        fiber_dir = np.stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ], axis=1)

        # --- Forward Model (Generazione Segnale) ---
        # Prodotto scalare gradienti-fibre
        cos_angles = np.dot(fiber_dir, self.bvecs.T)
        
        # Diffusività Apparente Fibra
        d_app_fiber = d_rad[:, None] + (d_ax[:, None] - d_rad[:, None]) * (cos_angles**2)
        
        # Componenti di Segnale
        sig_fib = np.exp(-self.bvals[None, :] * d_app_fiber)
        sig_res = np.exp(-self.bvals[None, :] * d_res[:, None])
        sig_hin = np.exp(-self.bvals[None, :] * d_hin[:, None])
        sig_wat = np.exp(-self.bvals[None, :] * d_wat)
        
        # Somma pesata
        clean_signal = (
            f_fiber[:, None] * sig_fib +
            f_res[:, None] * sig_res +
            f_hin[:, None] * sig_hin +
            f_wat[:, None] * sig_wat
        )

        # Aggiunta Rumore Rician
        if snr > 0:
            sigma = 1.0 / snr
            noise_real = np.random.normal(0, sigma, clean_signal.shape)
            noise_imag = np.random.normal(0, sigma, clean_signal.shape)
            noisy_signal = np.sqrt((clean_signal + noise_real)**2 + noise_imag**2)
        else:
            noisy_signal = clean_signal
        
        # Normalizzazione S0
        # Usiamo la media dei b<50 per robustezza
        b0_mean = np.mean(noisy_signal[:, self.bvals < 50], axis=1, keepdims=True) + 1e-9
        noisy_signal = noisy_signal / b0_mean

        # Tensori PyTorch
        X = torch.tensor(noisy_signal, dtype=torch.float32)
        # Labels Order: [Restricted, Hindered, Water, Fiber]
        Y = torch.tensor(np.stack([f_res, f_hin, f_wat, f_fiber], axis=1), dtype=torch.float32)

        return X, Y

# ==========================================
# 2. RETE NEURALE
# ==========================================
class DBSI_PriorNet(nn.Module):
    def __init__(self, n_input_meas: int):
        super().__init__()
        # Architettura ottimizzata per regressione fisica
        self.net = nn.Sequential(
            nn.Linear(n_input_meas, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            
            nn.Linear(128, 4), # Output: 4 Frazioni
            nn.Softmax(dim=1)  # Vincolo somma a 1
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. REGOLARIZZATORE (Wrapper)
# ==========================================
class DBSI_DeepRegularizer:
    """
    Gestisce il ciclo di vita del modello DL:
    1. Generazione dati sintetici specifici per protocollo e SNR.
    2. Training.
    3. Predizione sul volume reale.
    """
    def __init__(self, bvals: np.ndarray, bvecs: np.ndarray, epochs: int = 50, 
                 batch_size: int = 1024, lr: float = 1e-3, 
                 d_res_threshold: float = 0.3e-3):
        
        self.bvals = bvals
        self.bvecs = bvecs
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = None
        # Inizializza generatore con la soglia specifica
        self.generator = DBSISyntheticGenerator(bvals, bvecs, d_res_threshold)

    def train_on_synthetic(self, n_samples=100000, target_snr=30.0):
        print(f"[DL-Reg] Generating {n_samples} Scenario-Based samples (SNR={target_snr:.1f})...")
        
        X, Y = self.generator.generate_batch(n_samples, snr=target_snr)
        
        dataset = TensorDataset(X, Y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = DBSI_PriorNet(n_input_meas=len(self.bvals)).to(DEVICE)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.HuberLoss() # Robusto agli outlier
        
        self.model.train()
        print(f"[DL-Reg] Training Prior Network on {DEVICE}...")
        
        pbar = tqdm(range(self.epochs), desc="Training Priors", file=sys.stdout)
        for _ in pbar:
            epoch_loss = 0
            for bx, by in loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                preds = self.model(bx)
                loss = criterion(preds, by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            pbar.set_postfix({'loss': epoch_loss / len(loader)})

    def predict_volume(self, volume: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
        if self.model is None: raise RuntimeError("Model not trained.")
        
        self.model.eval()
        X_dim, Y_dim, Z_dim, N_meas = volume.shape
        valid_signals = volume[mask]
        
        # Normalizzazione dati reali
        b0_idx = self.bvals < 50
        if np.sum(b0_idx) > 0:
            s0 = np.mean(valid_signals[:, b0_idx], axis=1, keepdims=True)
            valid_signals = valid_signals / (s0 + 1e-9)
        
        # Batch Inference
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