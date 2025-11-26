# dbsi_toolbox/numba_backend.py
import numpy as np
from numba import njit, prange

@njit(fastmath=True, cache=True)
def build_design_matrix_numba(bvals, bvecs, iso_diffusivities, axial_diff, radial_diff):
    """Costruisce la matrice di design A (N_misure x N_basi) in modo vettorializzato."""
    n_meas = len(bvals)
    n_aniso = len(bvecs)
    n_iso = len(iso_diffusivities)
    n_cols = n_aniso + n_iso
    
    A = np.zeros((n_meas, n_cols), dtype=np.float64)
    
    # 1. Basi Anisotrope
    for j in range(n_aniso):
        norm = np.sqrt(bvecs[j, 0]**2 + bvecs[j, 1]**2 + bvecs[j, 2]**2)
        if norm > 0:
            fx, fy, fz = bvecs[j, 0]/norm, bvecs[j, 1]/norm, bvecs[j, 2]/norm
        else:
            fx, fy, fz = 0.0, 0.0, 0.0
            
        for i in range(n_meas):
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
def coordinate_descent_nnls(AtA, Aty, max_iter=300, tol=1e-7):
    """Solver NNLS ottimizzato per matrici dense di piccole dimensioni."""
    n_vars = AtA.shape[0]
    x = np.zeros(n_vars, dtype=np.float64)
    
    for _ in range(max_iter):
        max_change = 0.0
        for i in range(n_vars):
            dot_prod = 0.0
            for j in range(n_vars):
                dot_prod += AtA[i, j] * x[j]
            
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
def fit_volume_numba(data_flat, bvals, A, reg_lambda, mask_flat, n_aniso, idx_res_end, idx_hin_end):
    """Esegue il fitting su tutto il volume in parallelo su CPU."""
    n_voxels, n_meas = data_flat.shape
    n_bases = A.shape[1]
    
    # Pre-calcolo matrici (A^T * A) con regolarizzazione Tikhonov
    AtA = np.dot(A.T, A)
    if reg_lambda > 0:
        for i in range(n_bases):
            AtA[i, i] += reg_lambda
            
    results = np.zeros((n_voxels, 5), dtype=np.float32)
    
    for i in prange(n_voxels):
        if not mask_flat[i]:
            continue
            
        signal = data_flat[i, :]
        
        # Normalizzazione S0 robusta
        s0 = 0.0
        count_b0 = 0
        for k in range(n_meas):
            if bvals[k] < 50:
                s0 += signal[k]
                count_b0 += 1
        
        if count_b0 > 0: s0 /= count_b0
        if s0 <= 1e-6: continue
            
        y = np.empty(n_meas, dtype=np.float64)
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
        
        for j in range(n_aniso):
            f_fiber += weights[j]
            
        for j in range(n_aniso, n_bases):
            iso_idx = j - n_aniso
            w_val = weights[j]
            if iso_idx < idx_res_end: f_res += w_val
            elif iso_idx < idx_hin_end: f_hin += w_val
            else: f_wat += w_val
        
        total_w = f_fiber + f_res + f_hin + f_wat
        if total_w > 0:
            results[i, 0] = f_fiber / total_w
            results[i, 1] = f_res / total_w
            results[i, 2] = f_hin / total_w
            results[i, 3] = f_wat / total_w
        

    return results