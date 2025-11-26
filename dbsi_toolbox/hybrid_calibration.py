# dbsi_toolbox/hybrid_calibration.py

import numpy as np
import torch
import sys
from tqdm import tqdm
from .nlls_tensor_fit_dl import DBSI_TensorFit_DL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def optimize_hybrid_alpha(
    dl_solver, 
    bvals, 
    bvecs, 
    snr_estimate, 
    alphas=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0], # Range logaritmico
    n_monte_carlo=500, # 500 voxel sono statisticamente sufficienti e veloci
    d_res_threshold=0.3e-3
):
    """
    Trova l'alpha ottimale simulando il processo di fitting su dati sintetici
    con lo stesso livello di rumore dei dati reali.
    
    Returns:
        best_alpha (float): Il valore che minimizza l'errore rispetto alla Ground Truth.
    """
    print(f"\n[Auto-Calibration] Calibrazione Alpha (Monte Carlo: {n_monte_carlo} iter, SNR: {snr_estimate:.1f})...")
    
    # 1. Genera Ground Truth (Verità) e Segnale Rumoroso (Simulazione Paziente)
    # Usiamo il generatore interno del solver DL che è già configurato per il protocollo
    X_noisy_tensor, Y_true_tensor = dl_solver.generator.generate_batch(n_monte_carlo, snr=snr_estimate)
    
    # Dati per il fitter
    signals_noisy = X_noisy_tensor.numpy()
    # Y_true è [f_res, f_hin, f_wat, f_fib] -> La nostra Verità Assoluta
    fractions_true = Y_true_tensor.numpy() 
    
    # 2. Calcola i Priors DL (Cosa direbbe la rete su questi dati rumorosi?)
    dl_solver.model.eval()
    with torch.no_grad():
        inputs = X_noisy_tensor.to(next(dl_solver.model.parameters()).device)
        priors_pred = dl_solver.model(inputs).cpu().numpy()
    
    # 3. Torneo degli Alpha
    errors = {}
    
    # Disabilita la barra interna se siamo in un contesto nested, o usa file=sys.stdout
    for alpha in alphas:
        fitter = DBSI_TensorFit_DL(alpha_reg=alpha, d_res_threshold=d_res_threshold)
        fitter.current_bvecs = bvecs
        
        total_mae = 0.0 # Mean Absolute Error accumulato
        valid_samples = 0
        
        for i in range(n_monte_carlo):
            sig = signals_noisy[i]
            prior = priors_pred[i]
            true_fracs = fractions_true[i]
            
            try:
                # Fit Ibrido
                res = fitter.fit_voxel(sig, bvals, initial_guess=None, dl_priors=prior)
                
                # Confrontiamo con la VERITÀ (Y_true), non con i priors!
                est_fracs = np.array([
                    res.f_restricted, 
                    res.f_hindered, 
                    res.f_water, 
                    res.fiber_density
                ])
                
                # Errore medio assoluto sulle 4 frazioni
                mae = np.mean(np.abs(est_fracs - true_fracs))
                total_mae += mae
                valid_samples += 1
            except:
                pass # Ignora fallimenti catastrofici nel calcolo della media (rari)
            
        if valid_samples > 0:
            avg_error = total_mae / valid_samples
            errors[alpha] = avg_error
            # print(f"   Alpha {alpha:<4} -> MAE Error: {avg_error:.5f}") # Decommenta per debug
    
    # 4. Selezione Vincitore
    best_alpha = min(errors, key=errors.get)
    
    print(f"   -> Alpha Vincitore: {best_alpha} (Errore Medio: {errors[best_alpha]:.4f})")
    
    return best_alpha