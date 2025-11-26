import numpy as np
from numba import njit, prange

@njit(fastmath=True, cache=True)
def build_design_matrix_numba(bvals, bvecs, iso_diffusivities, axial_diff, radial_diff):
    """
    Costruisce la matrice di design A (N_misure x N_basi) usando Numba.
    """
    n_meas = len(bvals)
    n_aniso = len(bvecs)
    n_iso = len(iso_diffusivities)
    n_cols = n_aniso + n_iso
    
    A = np.zeros((n_meas, n_cols), dtype=np.float64)
    
    # 1. Basi Anisotrope (Fibre)
    for j in range(n_aniso):
        # Normalizzazione vettore (gestione divisione per zero interna)
        fib_norm = np.sqrt(bvecs[j, 0]**2 + bvecs[j, 1]**2 + bvecs[j, 2]**2)
        if fib_norm > 0:
            fx, fy, fz = bvecs[j, 0]/fib_norm, bvecs[j, 1]/fib_norm, bvecs[j, 2]/fib_norm
        else:
            fx, fy, fz = 0.0, 0.0, 0.0
            
        for i in range(n_meas):
            # Prodotto scalare bvec_meas . bvec_fiber
            cos_angle = (bvecs[i, 0]*fx + bvecs[i, 1]*fy + bvecs[i, 2]*fz)
            
            # Modello DBSI Anisotropo: D_rad + (D_ax - D_rad) * cos^2(theta)
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
    Solver NNLS velocissimo basato su Coordinate Descent.
    Risolve min ||Ax - y||^2 s.t. x >= 0
    Equivalente a risolvere min 0.5 * x^T (AtA) x - (Aty)^T x
    """
    n_vars = AtA.shape[0]
    x = np.zeros(n_vars, dtype=np.float64)
    
    for _ in range(max_iter):
        max_change = 0.0
        
        for i in range(n_vars):
            # Calcolo gradiente parziale
            # grad_i = (AtA * x)_i - Aty_i
            dot_prod = 0.0
            for j in range(n_vars):
                dot_prod += AtA[i, j] * x[j]
            
            # Passo di aggiornamento ottimale (se non vincolato)
            # x_new = x_i - (dot_prod - Aty[i]) / AtA[i, i]
            # Semplificato: x_new = (Aty[i] - sum_{j!=i} AtA[ij]x[j]) / AtA[ii]
            
            # Possiamo scriverlo come update basato sul residuo
            w = dot_prod - Aty[i]
            
            if AtA[i, i] > 1e-12:
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
def fit_volume_numba(data_flat, bvals, bvecs, A, reg_lambda, mask_flat):
    """
    Elabora un intero volume (appiattito) in parallelo.
    Sostituisce il ciclo lento in Python.
    """
    n_voxels, n_meas = data_flat.shape
    n_bases = A.shape[1]
    
    # Pre-calcolo matrici per NNLS (AtA e Aty)
    # Se usiamo regolarizzazione Tikhonov: (A'A + lambda*I)x = A'y
    AtA = np.dot(A.T, A)
    
    # Aggiungi regolarizzazione sulla diagonale
    if reg_lambda > 0:
        for i in range(n_bases):
            AtA[i, i] += reg_lambda
            
    # Array di output pre-allocati
    # 0: f_fiber, 1: f_restricted, 2: f_hindered, 3: f_water, 4: r2
    results = np.zeros((n_voxels, 5), dtype=np.float32)
    
    # Indice dove finiscono le basi anisotrope
    n_aniso = len(bvecs)
    
    # Indici per le frazioni isotrope (basato su soglie standard DBSI)
    # Assumiamo che A sia stato costruito con basi isotrope ordinate
    # Questo mapping andrebbe passato come argomento per massima generalità,
    # ma qui lo semplifichiamo per performance.
    
    # Parallel Loop
    for i in prange(n_voxels):
        if not mask_flat[i]:
            continue
            
        signal = data_flat[i, :]
        
        # Check validità segnale
        s0 = 0.0
        count_b0 = 0
        for k in range(n_meas):
            if bvals[k] < 50:
                s0 += signal[k]
                count_b0 += 1
        
        if count_b0 > 0:
            s0 /= count_b0
        
        if s0 <= 1e-6:
            continue
            
        # Normalizzazione
        y = np.zeros(n_meas)
        for k in range(n_meas):
            y[k] = signal[k] / s0
            
        # Preparazione NNLS: Aty = A.T * y
        Aty = np.dot(A.T, y)
        
        # Risoluzione
        weights = coordinate_descent_nnls(AtA, Aty)
        
        # --- Estrazione Metriche ---
        # Somma pesi anisotropi (Fibre Fraction)
        f_fiber = 0.0
        for j in range(n_aniso):
            f_fiber += weights[j]
            
        # Somma pesi isotropi (Restricted, Hindered, Water)
        # Nota: Qui servirebbe conoscere i bin esatti. 
        # Assumiamo per questo esempio che le ultime colonne siano ordinate per diffusività.
        # Bisognerebbe passare gli indici `idx_res`, `idx_hin`, `idx_wat` alla funzione.
        # Per ora salviamo la somma totale isotropa come esempio.
        f_iso_total = 0.0
        for j in range(n_aniso, n_bases):
            f_iso_total += weights[j]
            
        # Calcolo R2 (semplificato)
        # y_pred = A * weights
        ss_res = 0.0
        ss_tot = 0.0
        y_mean = 0.0
        for k in range(n_meas):
            y_mean += y[k]
        y_mean /= n_meas
        
        for k in range(n_meas):
            pred = 0.0
            for j in range(n_bases):
                pred += A[k, j] * weights[j]
            
            ss_res += (y[k] - pred)**2
            ss_tot += (y[k] - y_mean)**2
            
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        results[i, 0] = f_fiber
        results[i, 1] = f_iso_total # Placeholder, va diviso
        results[i, 4] = r2

    return results