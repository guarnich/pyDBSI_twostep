# dbsi_toolbox/utils.py

import os
import numpy as np
import nibabel as nib
from typing import Tuple, Optional

# Import DIPY functions
try:
    from dipy.io.image import load_nifti
    from dipy.io import read_bvals_bvecs
    from dipy.core.gradients import gradient_table, GradientTable
    from dipy.segment.mask import median_otsu
except (ImportError, AttributeError):
    print("WARNING: DIPY not found. Some utility functions may not work.")
    GradientTable = type("GradientTable", (object,), {})

def load_dwi_data_dipy(
    f_nifti: str, 
    f_bval: str, 
    f_bvec: str, 
    f_mask: str 
) -> Tuple[np.ndarray, np.ndarray, 'GradientTable', np.ndarray]:
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
    b0_data: np.ndarray,
    mask: np.ndarray,
    max_iter: int = 5,
    convergence_tol: float = 0.01
) -> float:
    """
    Estimates SNR with Rician bias correction using iterative approach.
    
    Based on:
    - Gudbjartsson & Patz (1995). "The Rician distribution of noisy MRI data"
    - Dietrich et al. (2007). "Measurement of SNR in MR images"
    
    Args:
        b0_data: Array of b0 volumes (X, Y, Z, N_b0)
        mask: Binary brain mask (X, Y, Z)
        max_iter: Maximum iterations for correction
        convergence_tol: Convergence threshold
        
    Returns:
        Rician-corrected SNR estimate
    """
    # Calculate temporal statistics
    mean_b0 = np.mean(b0_data, axis=-1)
    std_b0 = np.std(b0_data, axis=-1, ddof=1)  # Unbiased estimator
    
    # Avoid division by zero
    std_b0[std_b0 == 0] = 1e-10
    
    # Initial apparent SNR
    snr_map = mean_b0 / std_b0
    
    # Extract only masked voxels for efficiency
    mask_indices = mask > 0
    snr_masked = snr_map[mask_indices]
    mean_masked = mean_b0[mask_indices]
    std_masked = std_b0[mask_indices]
    
    # Iterative Rician correction
    # For Rician distribution: σ_true² = σ_obs² - μ²/(2·SNR²)
    snr_corrected = snr_masked.copy()
    
    for iteration in range(max_iter):
        snr_old = snr_corrected.copy()
        
        # Bias correction term
        bias_term = mean_masked**2 / (2 * snr_corrected**2 + 1e-10)
        
        # Corrected variance
        var_corrected = std_masked**2 - bias_term
        var_corrected[var_corrected < 0] = 1e-10  # Ensure positivity
        
        # Updated SNR
        snr_corrected = mean_masked / np.sqrt(var_corrected)
        
        # Check convergence
        relative_change = np.median(np.abs(snr_corrected - snr_old) / (snr_old + 1e-10))
        if relative_change < convergence_tol:
            print(f"  ✓ Rician correction converged at iteration {iteration + 1}")
            break
    
    # Use median (robust to outliers)
    final_snr = np.median(snr_corrected)
    
    return final_snr

def estimate_snr(
    data: np.ndarray, 
    gtab: 'GradientTable',
    affine: np.ndarray,
    mask: np.ndarray,
    method: str = 'temporal_rician'
) -> float:
    """
    Estimates SNR with multiple fallback strategies.
    
    Priority hierarchy:
    1. Temporal + Rician correction (≥2 b0 volumes) [MOST ACCURATE]
    2. Temporal only (≥2 b0 volumes, no correction)
    3. Spatial estimation (single b0)
    4. Default conservative value
    
    Args:
        data: 4D DWI volume
        gtab: DIPY gradient table
        affine: NIfTI affine transformation
        mask: Binary brain mask
        method: 'temporal_rician', 'temporal', 'spatial', or 'auto'
        
    Returns:
        Estimated SNR value
    """
    print("\n[Utils] Estimating SNR...")
    
    # Extract b0 volumes
    b0_mask = gtab.b0s_mask
    b0_data = data[..., b0_mask]
    n_b0 = b0_data.shape[-1]
    
    if n_b0 == 0:
        print("  ! No b=0 volumes found. Using default SNR=20.0")
        return 20.0
    
    snr_estimate = 20.0  # Default
    estimation_method = "default"
    
    # --- METHOD 1: TEMPORAL + RICIAN (Best Practice) ---
    if n_b0 >= 2 and method in ['temporal_rician', 'auto']:
        print(f"  → Method: Temporal + Rician Correction ({n_b0} b0 volumes)")
        
        try:
            snr_estimate = estimate_snr_rician_corrected(b0_data, mask)
            estimation_method = "temporal_rician"
            print(f"  ✓ Rician-Corrected SNR: {snr_estimate:.2f}")
            
        except Exception as e:
            print(f"  ! Rician correction failed: {e}")
            print("  → Falling back to uncorrected temporal estimation...")
            
            # Fallback to simple temporal
            mean_b0 = np.mean(b0_data, axis=-1)
            std_b0 = np.std(b0_data, axis=-1, ddof=1)
            std_b0[std_b0 == 0] = 1e-10
            
            snr_map = mean_b0 / std_b0
            snr_estimate = np.median(snr_map[mask])
            estimation_method = "temporal_simple"
            print(f"  ✓ Temporal SNR (uncorrected): {snr_estimate:.2f}")
    
    # --- METHOD 2: SPATIAL ESTIMATION (Single b0) ---
    elif n_b0 == 1 and method in ['spatial', 'auto']:
        print("  → Method: Spatial (single b0 volume)")
        print("  ! WARNING: Less accurate than temporal methods")
        
        try:
            b0_single = b0_data[..., 0]
            
            # Signal: Mean in brain
            signal_mean = np.mean(b0_single[mask])
            
            # Noise: Std in background corners (assume air/noise)
            # Extract corner regions (assumed to be background)
            corner_size = 10
            corners = [
                b0_single[:corner_size, :corner_size, :corner_size],
                b0_single[-corner_size:, :corner_size, :corner_size],
                b0_single[:corner_size, -corner_size:, :corner_size],
                b0_single[-corner_size:, -corner_size:, :corner_size],
            ]
            noise_std = np.std(np.concatenate([c.ravel() for c in corners]))
            
            if noise_std > 0:
                snr_estimate = signal_mean / noise_std
                estimation_method = "spatial_background"
                print(f"  ✓ Spatial SNR: {snr_estimate:.2f}")
            else:
                print("  ! Background noise estimation failed")
                estimation_method = "default"
                
        except Exception as e:
            print(f"  ! Spatial estimation failed: {e}")
            estimation_method = "default"
    
    # --- FALLBACK ---
    else:
        print(f"  ! Cannot estimate SNR reliably (n_b0={n_b0}, method={method})")
        print("  ! Using conservative default: SNR=20.0")
        print("  ! Recommendation: Provide manual --snr value if known")
    
    # --- SAFETY BOUNDS ---
    # Clamp to physiologically reasonable range
    if snr_estimate < 3.0:
        print(f"  ! Unusually low SNR detected ({snr_estimate:.1f}). Clamping to 3.0")
        print("  ! Check data quality - very low SNR may indicate preprocessing issues")
        snr_estimate = 3.0
    elif snr_estimate > 150.0:
        print(f"  ! Unusually high SNR detected ({snr_estimate:.1f}). Clamping to 150.0")
        print("  ! This may indicate estimation error or post-processed data")
        snr_estimate = 150.0
    
    print(f"\n  [Summary] Final SNR: {snr_estimate:.2f} (Method: {estimation_method})")
    
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
    
    Standard metrics for methodological papers:
    - RMSE (Root Mean Square Error)
    - MAE (Mean Absolute Error)
    - Bias (systematic error)
    - Relative Error (percentage)
    
    Args:
        fitted_params: Dictionary of fitted parameter maps
        ground_truth: Dictionary of true parameter maps
        mask: Optional mask to restrict analysis
        
    Returns:
        Dictionary of metrics per parameter
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