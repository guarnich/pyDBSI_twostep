import numpy as np
import torch
from tqdm import tqdm
from .nlls_tensor_fit_dl import DBSI_TensorFit_DL

def optimize_hybrid_alpha(
    dl_solver, 
    bvals, 
    bvecs, 
    snr_estimate, 
    alphas=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0],
    n_monte_carlo=500,
    d_res_threshold=0.3e-3
):
    """
    Trova l'alpha ottimale simulando il processo di fitting su dati sintetici
    con lo stesso livello di rumore dei dati reali.
    
    Returns:
        best_alpha (float): Il valore che minimizza l'errore di ricostruzione delle frazioni.
        results (dict): Statistiche per ogni alpha testato.
    """
    print(f"\n[Auto-Calibration] Calibrazione Alpha (Monte Carlo: {n_monte_carlo} iter, SNR: {snr_estimate:.1f})...")
    
    # 1. Genera Ground Truth (Verità) e Segnale Rumoroso
    # Usiamo il generatore interno del solver DL che è già configurato per il protocollo
    # Generiamo un batch di test
    X_noisy, Y_true = dl_solver.generator.generate_batch(n_monte_carlo, snr=snr_estimate)
    
    # Convertiamo in numpy per il fitter
    signals_noisy = X_noisy.numpy()
    fractions_true = Y_true.numpy() # [f_res, f_hin, f_wat, f_fib]
    
    # 2. Calcola i Priors DL (La "guida")
    # La rete vede il segnale rumoroso e fa la sua predizione
    dl_solver.model.eval()
    with torch.no_grad():
        # Passiamo il tensore alla rete (su GPU se disponibile)
        inputs = X_noisy.to(dl_solver.model.net[0].weight.device)
        priors_pred = dl_solver.model(inputs).cpu().numpy()
    
    # 3. Loop sugli Alpha
    errors = {}
    
    for alpha in tqdm(alphas, desc="Testing Alphas"):
        fitter = DBSI_TensorFit_DL(alpha_reg=alpha, d_res_threshold=d_res_threshold)
        fitter.current_bvecs = bvecs
        
        total_error = 0.0
        
        for i in range(n_monte_carlo):
            sig = signals_noisy[i]
            prior = priors_pred[i]
            true_fracs = fractions_true[i]
            
            # Fit Ibrido
            res = fitter.fit_voxel(sig, bvals, initial_guess=None, dl_priors=prior)
            
            # Calcolo Errore (MAE - Mean Absolute Error sulle frazioni)
            # Confrontiamo il risultato del fit con la VERITÀ FISICA (non con i priors!)
            est_fracs = np.array([res.f_restricted, res.f_hindered, res.f_water, res.fiber_density])
            
            # Errore medio su tutte le 4 componenti
            mae = np.mean(np.abs(est_fracs - true_fracs))
            total_error += mae
            
        avg_error = total_error / n_monte_carlo
        errors[alpha] = avg_error
    
    # 4. Selezione Vincitore
    best_alpha = min(errors, key=errors.get)
    
    print("\n--- Risultati Calibrazione ---")
    for a, e in errors.items():
        mark = "*" if a == best_alpha else ""
        print(f"Alpha {a:<5} -> MAE Error: {e:.5f} {mark}")
    print(f"Winner: Alpha = {best_alpha}")
    
    return best_alpha, errors