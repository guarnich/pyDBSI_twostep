import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm.notebook import tqdm
import sys
import os
from dbsi_toolbox.utils import load_dwi_data_dipy, save_parameter_maps

# Rileva GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 1. DEFINIZIONE CLASSI (Self-Supervised - CORRETTO)
# =============================================================================

class DBSI_PhysicsLayer(nn.Module):
    """
    Layer "Fisico" non addestrabile: implementa l'equazione DBSI.
    Converte i parametri (Frazioni, Diffusività) in Segnale DWI simulato.
    """
    def __init__(self, bvals, bvecs):
        super().__init__()
        self.register_buffer('bvals', torch.tensor(bvals, dtype=torch.float32))
        self.register_buffer('bvecs', torch.tensor(bvecs, dtype=torch.float32))

    def forward(self, params):
        # params: [batch, 10] 
        # Indici: 0=f_res, 1=d_res, 2=f_hin, 3=d_hin, 4=f_wat, 5=f_fib, 6=d_ax, 7=d_rad, 8=theta, 9=phi
        
        # --- 1. Frazioni (Softmax per somma=1) ---
        f_raw = params[:, [0, 2, 4, 5]] # res, hin, wat, fib
        f_norm = torch.softmax(f_raw, dim=1)
        f_res, f_hin, f_wat, f_fib = f_norm[:,0], f_norm[:,1], f_norm[:,2], f_norm[:,3]
        
        # --- 2. Diffusività (Sigmoide per range fisici) ---
        d_res = torch.sigmoid(params[:, 1]) * 0.3e-3          # 0 - 0.3
        d_hin = torch.sigmoid(params[:, 3]) * 2.2e-3 + 0.3e-3 # 0.3 - 2.5
        d_wat = torch.tensor(3.0e-3, device=DEVICE)           # Fisso
        
        d_ax = torch.sigmoid(params[:, 6]) * 3.0e-3           # 0 - 3.0
        d_rad = torch.sigmoid(params[:, 7]) * d_ax            # < d_ax (anisotropia)
        
        # --- 3. Angoli ---
        theta = params[:, 8]
        phi = params[:, 9]

        # Calcolo Direzione Fibra
        fib_dir = torch.stack([
            torch.sin(theta)*torch.cos(phi),
            torch.sin(theta)*torch.sin(phi),
            torch.cos(theta)
        ], dim=1)

        # --- 4. Forward Model ---
        cos_angles = torch.matmul(fib_dir, self.bvecs.T)
        d_app_fib = d_rad[:,None] + (d_ax[:,None] - d_rad[:,None]) * (cos_angles**2)
        
        # Segnale Ricostruito
        S = (f_res[:,None] * torch.exp(-self.bvals * d_res[:,None]) +
             f_hin[:,None] * torch.exp(-self.bvals * d_hin[:,None]) +
             f_wat[:,None] * torch.exp(-self.bvals * d_wat) +
             f_fib[:,None] * torch.exp(-self.bvals * d_app_fib))
             
        # RETURN: 2 Valori (Segnale, Frazioni)
        return S, f_norm

class DBSI_Autoencoder(nn.Module):
    def __init__(self, n_meas, bvals, bvecs):
        super().__init__()
        
        # ENCODER: Segnale Reale -> Parametri Latenti
        self.encoder = nn.Sequential(
            nn.Linear(n_meas, 256),
            nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Linear(128, 10) # 10 Parametri fisici
        )
        
        # DECODER: Parametri -> Segnale Ricostruito (via Fisica)
        self.decoder = DBSI_PhysicsLayer(bvals, bvecs)

    def forward(self, x):
        params = self.encoder(x)
        # Il decoder restituisce 2 valori
        signal_pred, fractions = self.decoder(params)
        # L'autoencoder ne restituisce 3 per comodità nel training
        return signal_pred, fractions, params

class SelfSupervisedSolver:
    def __init__(self, bvals, bvecs, lr=1e-3):
        self.bvals = bvals
        self.bvecs = bvecs
        self.model = DBSI_Autoencoder(len(bvals), bvals, bvecs).to(DEVICE)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
    def train_on_real_data(self, data_volume, mask, epochs=50, batch_size=4096):
        print(f"[Self-Supervised] Training on {np.sum(mask)} real voxels...")
        
        valid_data = data_volume[mask]
        b0_idx = self.bvals < 50
        if np.sum(b0_idx) > 0:
            b0_mean = np.mean(valid_data[:, b0_idx], axis=1, keepdims=True)
            valid_data = valid_data / (b0_mean + 1e-9)
        
        dataset = TensorDataset(torch.tensor(valid_data, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        loss_fn = nn.MSELoss()
        
        pbar = tqdm(range(epochs), desc="Training")
        for epoch in pbar:
            epoch_loss = 0
            for batch in loader:
                real_signal = batch[0].to(DEVICE)
                
                self.optimizer.zero_grad()
                
                # Qui spacchettiamo 3 valori perché chiamiamo self.model()
                recon_signal, fractions, params = self.model(real_signal)
                
                rec_loss = loss_fn(recon_signal, real_signal)
                entropy = -torch.mean(torch.sum(fractions * torch.log(fractions + 1e-9), dim=1))
                
                total_loss = rec_loss + 0.001 * entropy
                
                total_loss.backward()
                self.optimizer.step()
                epoch_loss += rec_loss.item()
            
            pbar.set_postfix({'MSE': epoch_loss/len(loader)})

    def predict_volume(self, data_volume, mask):
        self.model.eval()
        X, Y, Z, N = data_volume.shape
        
        valid_data = data_volume[mask]
        b0_mean = np.mean(valid_data[:, self.bvals<50], axis=1, keepdims=True) + 1e-9
        valid_data = valid_data / b0_mean
        
        dataset = TensorDataset(torch.tensor(valid_data, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=4096, shuffle=False)
        
        frac_preds = []
        with torch.no_grad():
            for batch in loader:
                # FIX: Qui chiamiamo self.model.decoder, che ritorna SOLO 2 valori
                # _, fracs = decoder(...)
                params = self.model.encoder(batch[0].to(DEVICE))
                _, fracs = self.model.decoder(params)
                frac_preds.append(fracs.cpu().numpy())
                
        flat_fracs = np.concatenate(frac_preds, axis=0)
        
        maps = {}
        keys = ['ssl_restricted_fraction', 'ssl_hindered_fraction', 'ssl_water_fraction', 'ssl_fiber_fraction']
        for i, k in enumerate(keys):
            vol = np.zeros((X, Y, Z))
            vol[mask] = flat_fracs[:, i]
            maps[k] = vol
            
        return maps