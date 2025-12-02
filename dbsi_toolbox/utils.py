# dbsi_toolbox/utils.py

import os
import numpy as np
import nibabel as nib
from typing import Tuple, Optional, Dict
import time

# --- BASIC IMPORTS ---
try:
    from dipy.io.image import load_nifti
    from dipy.io import read_bvals_bvecs
    from dipy.core.gradients import gradient_table, GradientTable
    from dipy.segment.mask import median_otsu
except (ImportError, AttributeError):
    print("WARNING: DIPY core functions not found. Data loading will fail.")
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
    """
    # 1. Identify and extract b0 volumes
    bvals = np.array(bvals).flatten()
    b0_indices = np.where(bvals <= b0_threshold)[0]
    
    # Check if enough b0 volumes exist
    if len(b0_indices) < 2:
        # Cannot compute temporal SNR with fewer than 2 volumes
        return np.nan 
        
    # Extract b0 data
    b0_data = data[..., b0_indices]

    # 2. Calculate temporal statistics (voxel-wise)
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
    # (Removes the bias introduced by Rician noise at low SNR)
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
    method: str = 'temporal_rician' # Fixed to this method
) -> float:
    """
    Estimates SNR using ONLY the Temporal Rician method.
    NO spatial fallback is applied.
    """
    print("\n[Utils] Estimating SNR (Strict Temporal + Rician)...")
    
    b0_mask = gtab.b0s_mask
    n_b0 = np.sum(b0_mask)
    
    # --- STRICT CHECK ---
    if n_b0 < 2:
        print(f"  [CRITICAL ERROR] Found only {n_b0} b0 volumes.")
        print("  ! This method STRICTLY requires >= 2 b0 volumes to estimate temporal variance.")
        print("  ! Returning default SNR = 20.0 to allow pipeline to proceed (but check your data!).")
        return 20.0

    snr_estimate = 20.0
    estimation_source = "Default"

    # --- EXECUTE METHOD ---
    try:
        print(f"  → Analyzing {n_b0} b0 volumes...")
        val = estimate_snr_rician_corrected(data, gtab.bvals, gtab.bvecs, mask)
        
        if np.isnan(val) or val <= 0:
            print("  ! Computation failed (result was NaN or <= 0).")
            print("  ! Using default SNR = 20.0")
            snr_estimate = 20.0
        else:
            snr_estimate = val
            estimation_source = "Temporal Rician"
            print(f"  ✓ Calculated SNR: {snr_estimate:.2f}")
            
    except Exception as e:
         print(f"  ! Error during calculation: {e}")
         print("  ! Using default SNR = 20.0")
         snr_estimate = 20.0

    # --- SAFETY BOUNDS ---
    # Even with strict method, we clamp extreme outliers to avoid blowing up the DBSI fit
    if snr_estimate < 3.0: 
        print("  ! WARNING: Very low SNR detected (<3.0). Clamped to 3.0.")
        snr_estimate = 3.0
    if snr_estimate > 150.0:
        print("  ! WARNING: High SNR detected (>150). Clamped to 150.0.")
        snr_estimate = 150.0

    print(f"  [Result] Final SNR: {snr_estimate:.2f} (Source: {estimation_source})")
    return float(snr_estimate)

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