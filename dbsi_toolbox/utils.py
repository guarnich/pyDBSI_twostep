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
    # Dummy class to prevent crashes if dipy is missing
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
    # --- STRICT INPUT CHECK ---
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

def estimate_snr(
    data: np.ndarray, 
    gtab: 'GradientTable', #type: ignore
    affine: np.ndarray,
    mask: np.ndarray
) -> float:
    """
    Estimates SNR focusing on Temporal Stability.
    
    Priority 1 (Temporal): If >= 2 b0s, calculate SNR voxel-wise over time.
                           This is the most robust method for multi-volume data.
    
    Priority 2 (Fallback): If only 1 b0 exists, we cannot reliably estimate temporal SNR.
                           Returns a default value and advises the user to provide a manual input.
    """
    print("\n[Utils] Estimating SNR...")
    
    # 1. Extract b0 volumes
    b0_mask = gtab.b0s_mask
    b0_data = data[..., b0_mask]
    n_b0 = b0_data.shape[-1]
    
    if n_b0 == 0:
        print("  ! No b=0 volumes. Defaulting to SNR=30.0")
        return 30.0

    snr_est = 0.0

    # --- METHOD 1: TEMPORAL (Primary) ---
    if n_b0 >= 2:
        print(f"  ✓ Method: Temporal (based on {n_b0} b=0 volumes)")
        
        # Mean and Standard Deviation across time (4th dimension)
        mean_b0 = np.mean(b0_data, axis=-1)
        std_b0 = np.std(b0_data, axis=-1)
        
        # Avoid division by zero
        std_b0[std_b0 == 0] = 1e-10
        
        # Voxel-wise SNR map
        snr_map = mean_b0 / std_b0
        
        # Extract median SNR ONLY within the brain mask
        if np.sum(mask) > 0:
            snr_est = np.median(snr_map[mask])
            print(f"  ✓ Median Temporal SNR (Brain): {snr_est:.2f}")
        else:
            print("  ! Mask is empty. Defaulting to 20.0")
            snr_est = 20.0

    # --- FALLBACK (Single b0) ---
    else:
        print("  ! SNR estimation not possible: only one b=0 volume detected.")
        print("  ! Temporal estimation requires at least 2 b=0 volumes.")
        print("  ! Please use a manually chosen value (e.g., via CLI input) if available.")
        print("  ! Defaulting to SNR = 20.0")
        snr_est = 20.0

    # Safety Clamping (to avoid extreme values breaking calibration)
    if snr_est < 5.0:
        print(f"  ! Low SNR detected ({snr_est:.1f}). Clamping to 5.0")
        snr_est = 5.0
    elif snr_est > 100.0:
        print(f"  ! Very High SNR detected ({snr_est:.1f}). Clamping to 100.0")
        snr_est = 100.0
        
    return float(snr_est)

def save_parameter_maps(param_maps, affine, output_dir, prefix='dbsi'):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[Utils] Saving {len(param_maps)} maps to: {output_dir}")
    
    for k, v in param_maps.items():
        try:
            # Save as float32 for compatibility
            nib.save(nib.Nifti1Image(v.astype(np.float32), affine), 
                     os.path.join(output_dir, f'{prefix}_{k}.nii.gz'))
        except Exception as e:
            print(f"  ! Error saving {k}: {e}")
            
    print("  ✓ Done.")