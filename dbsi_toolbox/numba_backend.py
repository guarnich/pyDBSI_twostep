# dbsi_toolbox/numba_backend.py
import numpy as np
from numba import njit, prange

@njit(fastmath=True, cache=True)
def build_design_matrix_numba(bvals, bvecs, iso_diffusivities, axial_diff, radial_diff):
    """
    Costruisce la matrice di design A (N_misure x N_basi).
    Assicurarsi che gli input siano C-contiguous.
    """
    n_meas = bvals.shape[0]
    n_aniso = bvecs.shape[0]
    n_iso = iso_diffusivities.shape[0]
    n_cols = n_aniso + n_iso
    
    A = np.zeros((n_meas, n_cols), dtype=np.float64)
    
    # 1. Basi Anisotrope
    for j in range(n_aniso):
        # Normalizzazione vettore bvec
        norm = np.sqrt(bvecs[j, 0]**2 + bvecs[j, 1]**2 + bvecs[j, 2]**2)
        if norm > 1e-9:
            fx = bvecs[j, 0] / norm
            fy = bvecs[j, 1] / norm
            fz = bvecs[j, 2] / norm
        else:
            fx, fy, fz = 0.0, 0.0, 0.0
            
        for i in range(n_meas):
            # Prodotto scalare tra gradiente applicato e direzione fibra
            cos_angle = (bvecs[i, 0]*fx + bvecs[i, 1]*fy + bvecs[i, 2]*fz)
            D_app = radial_diff + (axial_diff - radial_diff) * (cos_angle**2)
            A[i, j] = np.exp(-bvals[i] * D_app)
            
    # 2. Basi Isotrope
    for k in range(n_iso):
        D_iso = iso_diffusivities[k]
        col_idx = n_aniso + k
        for i in range(n_meas):
            A[i, col_idx] = np.exp(-bvals[i] * D_iso)
            
    return A

@njit(fastmath=True, cache=True)
def coordinate_descent_nnls(AtA, Aty, max_iter=500, tol=1e-8):
    """
    Solver NNLS ottimizzato (Coordinate Descent).
    """
    n_vars = AtA.shape[0]
    x = np.zeros(n_vars, dtype=np.float64)
    
    for _ in range(max_iter):
        max_change = 0.0
        for i in range(n_vars):
            # Calcolo gradiente parziale
            dot_prod = 0.0
            for j in range(n_vars):
                dot_prod += AtA[i, j] * x[j]
            
            w = dot_prod - Aty[i]
            
            # Aggiornamento se la diagonale è stabile
            if AtA[i, i] > 1e-15:
                delta = -w / AtA[i, i]
                x_new = max(0.0, x[i] + delta)
                change = np.abs(x_new - x[i])
                if change > max_change:
                    max_change = change
                x[i] = x_new
        
        if max_change < tol:
            break
    return x

@njit(parallel=True, fastmath=True)
def fit_volume_numba(data_flat, bvals, A, reg_lambda, mask_flat, n_aniso, idx_res_end, idx_hin_end):
    """
    Esegue il fitting su tutto il volume in parallelo.
    IMPORTANTE: data_flat DEVE essere C-contiguous (n_voxels, n_meas).
    """
    n_voxels = data_flat.shape[0]
    n_meas = data_flat.shape[1]
    n_bases = A.shape[1]
    
    # Pre-calcolo matrici (A^T * A) con regolarizzazione Tikhonov
    # Eseguito una volta sola fuori dal loop parallelo
    AtA = np.dot(A.T, A)
    if reg_lambda > 0:
        for i in range(n_bases):
            AtA[i, i] += reg_lambda
            
    # Array risultati: [f_fiber, f_res, f_hin, f_wat, r_squared]
    results = np.zeros((n_voxels, 5), dtype=np.float64)
    
    for i in prange(n_voxels):
        if not mask_flat[i]:
            continue
            
        # Copia locale del segnale (necessaria per evitare problemi di accesso memoria in parallelo)
        signal = data_flat[i, :]
        
        # Normalizzazione S0 robusta
        s0 = 0.0
        count_b0 = 0
        for k in range(n_meas):
            if bvals[k] < 50:
                s0 += signal[k]
                count_b0 += 1
        
        if count_b0 > 0: 
            s0 /= count_b0
        
        # Skip se segnale troppo basso o invalido
        if s0 <= 1e-6: 
            continue
            
        # Preparazione target y
        y = np.zeros(n_meas, dtype=np.float64)
        for k in range(n_meas):
            y[k] = signal[k] / s0
            
        # Preparazione e Risoluzione NNLS
        Aty = np.dot(A.T, y)
        weights = coordinate_descent_nnls(AtA, Aty)
        
        # Aggregazione Metriche
        f_fiber = 0.0
        f_res = 0.0
        f_hin = 0.0
        f_wat = 0.0
        
        # Somma pesi anisotropi
        for j in range(n_aniso):
            f_fiber += weights[j]
            
        # Somma pesi isotropi con slicing logico basato sugli indici precalcolati
        for j in range(n_aniso, n_bases):
            iso_idx = j - n_aniso
            w_val = weights[j]
            
            # La logica degli indici deve essere coerente con np.sum(condizione)
            if iso_idx < idx_res_end: 
                f_res += w_val
            elif iso_idx < idx_hin_end: # Nota: idx_hin_end è cumulativo nel chiamante? Sì.
                f_hin += w_val
            else:
                f_wat += w_val
        
        total_w = f_fiber + f_res + f_hin + f_wat
        
        # Calcolo R-quadro approssimato
        # Calcoliamo y_pred = A * weights
        ss_res = 0.0
        ss_tot = 0.0
        y_mean = 0.0
        
        # Calcolo media y
        for k in range(n_meas):
            y_mean += y[k]
        y_mean /= n_meas
        
        # Calcolo residui
        for k in range(n_meas):
            y_pred_k = 0.0
            for b in range(n_bases):
                y_pred_k += A[k, b] * weights[b]
            
            ss_res += (y[k] - y_pred_k)**2
            ss_tot += (y[k] - y_mean)**2
            
        r2 = 0.0
        if ss_tot > 1e-12:
            r2 = 1.0 - (ss_res / ss_tot)

        # Salvataggio risultati normalizzati
        if total_w > 1e-9:
            results[i, 0] = f_fiber / total_w
            results[i, 1] = f_res / total_w
            results[i, 2] = f_hin / total_w
            results[i, 3] = f_wat / total_w
            results[i, 4] = r2

    return results