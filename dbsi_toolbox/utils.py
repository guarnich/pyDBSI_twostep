# dbsi_toolbox/utils.py

import os
import numpy as np
import nibabel as nib
from typing import Tuple, Optional, Dict
import time

# Import DIPY functions
try:
    from dipy.io.image import load_nifti
    from dipy.io import read_bvals_bvecs
    from dipy.core.gradients import gradient_table, GradientTable
    from dipy.segment.mask import median_otsu
    
    # Advanced Noise Estimation Imports
    from dipy.denoise.noise_estimate import pca_noise_estimate, piesno
except (ImportError, AttributeError):
    print("WARNING: DIPY not found or incomplete. Some utility functions may not work.")
    GradientTable = type("GradientTable", (object,), {})

def load_dwi_data_dipy(
    f_nifti: str, 
    f_bval: str, 
    f_bvec: str, 
    f_mask: str 
) -> Tuple[np.ndarray, np.ndarray, 'GradientTable', np.ndarray]: #type: ignore
    """
    Loads DWI data, bvals, bvecs, and the MANDATORY brain mask.
    """
    if not f_mask:
        raise ValueError("\n[CRITICAL] Brain Mask is MISSING. Please provide it.")

    print(f"[Utils] Loading data from: {f_nifti}")
    data, affine = load_nifti(f_nifti)
    
    print(f"[Utils] Loading bvals/bvecs...")
    bvals, bvecs = read_bvals_bvecs(f_bval, f_bvec)
    gtab = gradient_table(bvals, bvecs=bvecs)
    
    print(f"[Utils] Loading mask...")
    mask_data, _ = load_nifti(f_mask)
    mask_data = mask_data.astype(bool)
    
    if mask_data.shape != data.shape[:3]:
        raise ValueError(f"Mask shape {mask_data.shape} mismatch data {data.shape[:3]}")
    
    return data, affine, gtab, mask_data

def estimate_snr_rician_corrected(
    data: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    mask: np.ndarray,
    b0_threshold: float = 50.0,
    max_iter: int = 5,
    convergence_tol: float = 0.01
) -> float:
    """
    Estimates SNR using Temporal Variance with Iterative Rician Bias Correction.
    (This is the 'classic' method).
    """
    # 1. Identify and extract b0 volumes
    bvals = np.array(bvals).flatten()
    b0_indices = np.where(bvals <= b0_threshold)[0]
    
    # Check if enough b0 volumes exist
    if len(b0_indices) < 2:
        return np.nan # Cannot compute
        
    # Extract b0 data
    b0_data = data[..., b0_indices]

    # 2. Calculate temporal statistics
    mean_b0 = np.mean(b0_data, axis=-1)
    std_b0 = np.std(b0_data, axis=-1, ddof=1)
    
    # Avoid division by zero
    std_b0[std_b0 == 0] = 1e-10
    
    # Initial apparent SNR
    snr_map = mean_b0 / std_b0
    
    # 3. Extract only masked voxels for efficiency
    mask_indices = mask > 0
    snr_masked = snr_map[mask_indices]
    mean_masked = mean_b0[mask_indices]
    std_masked = std_b0[mask_indices]
    
    # 4. Iterative Rician correction
    snr_corrected = snr_masked.copy()
    
    for iteration in range(max_iter):
        snr_old = snr_corrected.copy()
        
        # Bias correction term
        bias_term = mean_masked**2 / (2 * snr_corrected**2 + 1e-10)
        
        # Corrected variance
        var_corrected = std_masked**2 - bias_term
        var_corrected[var_corrected < 0] = 1e-10 
        
        # Updated SNR
        snr_corrected = mean_masked / np.sqrt(var_corrected)
        
        # Check convergence
        relative_change = np.median(np.abs(snr_corrected - snr_old) / (snr_old + 1e-10))
        if relative_change < convergence_tol:
            break
    
    return float(np.median(snr_corrected))

def estimate_snr(
    data: np.ndarray, 
    gtab: 'GradientTable', #type: ignore
    affine: np.ndarray,
    mask: np.ndarray,
    method: str = 'auto'
) -> float:
    """
    Runs comprehensive SNR analysis comparing multiple methods:
    1. MP-PCA (Marchenko-Pastur): Best for modern data, uses redundancy.
    2. Temporal Rician: Classic method, requires multiple b0s.
    3. PIESNO: Uses background noise, good if b0s are scarce.
    
    Returns:
        The SNR value determined by the selected 'method' (or the best one if 'auto').
    """
    print("\n[Utils] Estimating SNR (Comprehensive Analysis)...")
    
    b0_mask = gtab.b0s_mask
    n_b0 = np.sum(b0_mask)
    
    results = {}
    
    # =========================================================
    # 1. METHOD: MP-PCA (State of the Art)
    # =========================================================
    try:
        start = time.time()
        # MP-PCA estimates noise sigma voxel-wise using all volumes
        sigma_mppca = pca_noise_estimate(data, gtab, verify_is_rotation=False)
        
        # Calculate Signal (mean b0)
        if n_b0 > 0:
            S0 = np.mean(data[..., b0_mask], axis=-1)
        else:
            S0 = data[..., 0] # Fallback
            
        # SNR = Signal / Noise
        snr_map_mppca = S0 / (sigma_mppca + 1e-12)
        mppca_val = np.median(snr_map_mppca[mask])
        
        results['MP-PCA'] = {
            'value': mppca_val,
            'time': time.time() - start,
            'desc': 'Local PCA (Robust)'
        }
    except Exception as e:
        print(f"  ! MP-PCA skipped: {e}")
        results['MP-PCA'] = {'value': np.nan, 'desc': 'Failed'}

    # =========================================================
    # 2. METHOD: TEMPORAL RICIAN (Classic)
    # =========================================================
    try:
        start = time.time()
        temp_val = estimate_snr_rician_corrected(data, gtab.bvals, gtab.bvecs, mask)
        
        desc = f"Temporal ({n_b0} b0s)"
        if np.isnan(temp_val):
            desc += " - Insufficient b0s"
            
        results['Temporal'] = {
            'value': temp_val,
            'time': time.time() - start,
            'desc': desc
        }
    except Exception as e:
        results['Temporal'] = {'value': np.nan, 'desc': f'Error: {str(e)[:20]}'}

    # =========================================================
    # 3. METHOD: PIESNO (Background Noise)
    # =========================================================
    try:
        start = time.time()
        # PIESNO typically runs on a single volume (first b0)
        if n_b0 > 0:
            # Find index of first b0
            first_b0_idx = np.where(b0_mask)[0][0]
            b0_slice = data[..., first_b0_idx]
        else:
            b0_slice = data[..., 0]

        sigma_piesno = piesno(b0_slice, N=1, return_mask=False)
        
        # Signal in mask
        mean_signal = np.mean(b0_slice[mask])
        piesno_val = mean_signal / (sigma_piesno + 1e-12)
        
        results['PIESNO'] = {
            'value': piesno_val,
            'time': time.time() - start,
            'desc': 'Background Estimation'
        }
    except Exception as e:
        results['PIESNO'] = {'value': np.nan, 'desc': f'Error: {str(e)[:20]}'}

    # =========================================================
    # REPORTING & DECISION
    # =========================================================
    print(f"\n  {'-'*70}")
    print(f"  {'METHOD':<15} | {'SNR':<10} | {'TIME (s)':<10} | {'NOTES':<20}")
    print(f"  {'-'*70}")
    
    for name, res in results.items():
        val_str = f"{res['value']:.2f}" if not np.isnan(res['value']) else "N/A"
        time_str = f"{res.get('time', 0):.2f}"
        print(f"  {name:<15} | {val_str:<10} | {time_str:<10} | {res['desc']:<20}")
    print(f"  {'-'*70}")

    # Logic to choose the return value
    final_snr = 20.0 # Ultimate fallback
    selected_source = "default"

    # Hierarchy of trust if 'auto'
    if method in ['auto', 'compare', 'all']:
        if not np.isnan(results['MP-PCA']['value']) and results['MP-PCA']['value'] > 0:
            final_snr = results['MP-PCA']['value']
            selected_source = "MP-PCA"
        elif not np.isnan(results['Temporal']['value']) and results['Temporal']['value'] > 0:
            final_snr = results['Temporal']['value']
            selected_source = "Temporal"
        elif not np.isnan(results['PIESNO']['value']) and results['PIESNO']['value'] > 0:
            final_snr = results['PIESNO']['value']
            selected_source = "PIESNO"
            
    # Explicit selection
    elif method.lower() == 'mppca' and not np.isnan(results['MP-PCA']['value']):
        final_snr = results['MP-PCA']['value']
        selected_source = "MP-PCA"
    elif method.lower() == 'temporal' and not np.isnan(results['Temporal']['value']):
        final_snr = results['Temporal']['value']
        selected_source = "Temporal"
    elif method.lower() == 'spatial' or method.lower() == 'piesno':
         if not np.isnan(results['PIESNO']['value']):
            final_snr = results['PIESNO']['value']
            selected_source = "PIESNO"

    # Safety clamping
    if final_snr < 3.0: 
        print("  ! WARNING: Very low SNR detected. Clamped to 3.0.")
        final_snr = 3.0
    if final_snr > 150.0:
        print("  ! WARNING: Suspiciously high SNR. Clamped to 150.0.")
        final_snr = 150.0

    print(f"\n  ✓ Selected for Analysis: {final_snr:.2f} (Source: {selected_source})")
    return float(final_snr)

def save_parameter_maps(param_maps, affine, output_dir, prefix='dbsi'):
    """
    Saves parameter maps as NIfTI files with proper data types.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[Utils] Saving {len(param_maps)} maps to: {output_dir}")
    
    for k, v in param_maps.items():
        try:
            # Use float32 for space efficiency while maintaining precision
            img_data = v.astype(np.float32)
            
            # Create NIfTI with proper metadata
            nii = nib.Nifti1Image(img_data, affine)
            nii.header['descrip'] = f'DBSI {k.replace("_", " ").title()}'
            
            output_path = os.path.join(output_dir, f'{prefix}_{k}.nii.gz')
            nib.save(nii, output_path)
            
        except Exception as e:
            print(f"  ! Error saving {k}: {e}")
    
    print("  ✓ All maps saved successfully.")

def compute_validation_metrics(
    fitted_params: dict,
    ground_truth: dict,
    mask: Optional[np.ndarray] = None
) -> dict:
    """
    Computes validation metrics for comparison with ground truth.
    """
    metrics = {}
    
    for param_name in fitted_params.keys():
        if param_name not in ground_truth:
            continue
        
        fitted = fitted_params[param_name]
        truth = ground_truth[param_name]
        
        # Apply mask if provided
        if mask is not None:
            fitted = fitted[mask]
            truth = truth[mask]
        else:
            fitted = fitted.ravel()
            truth = truth.ravel()
        
        # Remove NaN/Inf
        valid = np.isfinite(fitted) & np.isfinite(truth)
        fitted = fitted[valid]
        truth = truth[valid]
        
        if len(fitted) == 0:
            continue
        
        # Compute metrics
        error = fitted - truth
        
        metrics[param_name] = {
            'rmse': float(np.sqrt(np.mean(error**2))),
            'mae': float(np.mean(np.abs(error))),
            'bias': float(np.mean(error)),
            'relative_error': float(np.mean(np.abs(error) / (np.abs(truth) + 1e-10))) * 100,
            'correlation': float(np.corrcoef(fitted, truth)[0, 1]) if len(fitted) > 1 else 0.0
        }
    
    return metrics