# dbsi_toolbox/utils.py

import os
import numpy as np
import nibabel as nib
from typing import Tuple, Optional

# Import DIPY functions for compatibility
try:
    from dipy.io.image import load_nifti
    from dipy.io import read_bvals_bvecs
    from dipy.core.gradients import gradient_table, GradientTable # type: ignore
except (ImportError, AttributeError):
    print("WARNING: DIPY not found. Some utility functions may not work.")
    print("Install with: pip install dipy")
    
    # Create a dummy type to avoid import errors if dipy isn't present
    GradientTable = type("GradientTable", (object,), {})


def load_dwi_data_dipy(
    f_nifti: str, 
    f_bval: str, 
    f_bvec: str, 
    f_mask: str 
) -> Tuple[np.ndarray, np.ndarray, 'GradientTable', Optional[np.ndarray]]: # type: ignore
    """
    Loads DWI data, bvals, bvecs, and an optional mask using DIPY.
    
    Args:
        f_nifti: Path to the 4D NIfTI file (.nii or .nii.gz)
        f_bval: Path to the .bval file
        f_bvec: Path to the .bvec file
        f_mask: Optional path to the 3D NIfTI mask file
        
    Returns:
        A tuple containing:
        - data (np.ndarray): 4D DWI data
        - affine (np.ndarray): Affine matrix
        - gtab (GradientTable): DIPY gradient table object
        - mask (np.ndarray | None): 3D boolean mask or None
    """
    print(f"[Utils] Loading data from: {f_nifti}")
    data, affine = load_nifti(f_nifti)
    
    print(f"[Utils] Loading bvals/bvecs from: {f_bval}, {f_bvec}")
    bvals, bvecs = read_bvals_bvecs(f_bval, f_bvec)
    
    # Pass 'bvecs' as keyword argument to avoid Warning
    gtab = gradient_table(bvals, bvecs=bvecs)
    
    print(f"  ✓ Volume: {data.shape}, Bvals: {len(gtab.bvals)}, Bvecs: {gtab.bvecs.shape}")
    print(f"  ✓ No. of b=0 volumes: {np.sum(gtab.b0s_mask)}")
    
    mask_data = None
    if f_mask:
        print(f"[Utils] Loading mask from: {f_mask}")
        mask_data, mask_affine = load_nifti(f_mask)
        mask_data = mask_data.astype(bool)
        
        # Validation
        if mask_data.shape != data.shape[:3]:
            raise ValueError(
                f"Mask shape {mask_data.shape} does not match "
                f"data shape {data.shape[:3]}"
            )
        print(f"  ✓ Mask: {mask_data.shape}, Voxels: {np.sum(mask_data):,}")
    else:
        print("  ! No mask provided.")
    
    return data, affine, gtab, mask_data


def estimate_snr(
    data: np.ndarray, 
    gtab: 'GradientTable', #type: ignore
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Estimates SNR (Signal-to-Noise Ratio) using b=0 images.
    
    Strategy:
    1. If >= 3 b=0 volumes: Use 'temporal' method (Voxel-wise Mean/Std).
    2. If < 3 b=0 volumes: Use 'spatial' method (Signal inside Mask / Noise outside Mask).
    
    Args:
        data: 4D DWI volume (X, Y, Z, N)
        gtab: DIPY GradientTable
        mask: 3D binary mask of the brain (optional but recommended)
        
    Returns:
        float: Estimated SNR.
    """
    print("\n[Utils] Automatically estimating SNR...")
    
    # 1. Extract b=0 volumes
    b0_mask = gtab.b0s_mask
    b0_data = data[..., b0_mask]
    n_b0 = b0_data.shape[-1]
    
    if n_b0 == 0:
        print("  ! WARNING: No b=0 volumes found. Returning default SNR = 30.0")
        return 30.0

    # Ensure mask is boolean if present
    if mask is not None:
        mask = mask.astype(bool)
        if np.sum(mask) == 0:
            print("  ! Empty mask. Returning default SNR = 30.0")
            return 30.0
    else:
        # Without mask, spatial SNR is impossible. Return default.
        print("  ! No mask provided for SNR estimation. Returning default = 30.0")
        return 30.0

    snr_est = 0.0

    # --- METHOD 1: Temporal SNR (if enough b0s) ---
    if n_b0 >= 3:
        print(f"  ✓ Method: Temporal (based on {n_b0} b=0 volumes)")
        # Calculate mean and std along the temporal dimension (4th dim)
        mean_b0 = np.mean(b0_data, axis=-1)
        std_b0 = np.std(b0_data, axis=-1)
        
        # Avoid division by zero
        std_b0[std_b0 == 0] = 1e-10
        
        # Voxel-wise SNR
        snr_map = mean_b0 / std_b0
        
        # Take median SNR only inside the brain mask
        snr_est = np.median(snr_map[mask])
        print(f"  ✓ Calculated SNR (Voxel-wise Median): {snr_est:.2f}")

    # --- METHOD 2: Spatial SNR (Signal/Background) ---
    else:
        print(f"  ✓ Method: Spatial (few b0s available: {n_b0})")
        # Use mean of all available b0s to reduce visual noise
        mean_b0 = np.mean(b0_data, axis=-1)
        
        # Signal: Mean intensity inside the provided mask
        signal_mean = np.mean(mean_b0[mask])
        
        # Noise: Standard deviation of everything OUTSIDE the mask
        background_mask = ~mask
        
        # Remove artifacts (true zeros from padding)
        noise_data = mean_b0[background_mask]
        noise_data = noise_data[noise_data > 0] 
        
        if len(noise_data) == 0:
             print("  ! Unable to find valid background noise. Returning default = 30.0")
             return 30.0
             
        noise_std = np.std(noise_data)
        
        # Correction for Rician/Rayleigh noise in magnitude images
        # Real_SD = Background_SD / 0.655
        noise_std_corrected = noise_std / 0.655
        
        snr_est = signal_mean / noise_std_corrected
        print(f"  ✓ Mean Signal (Inside Mask): {signal_mean:.2f}")
        print(f"  ✓ Noise Std (Outside Mask, corr): {noise_std_corrected:.2f}")
        print(f"  ✓ Calculated SNR: {snr_est:.2f}")

    # Safety limits (Sanity Check)
    if snr_est < 5.0:
        print("  ! Very low SNR detected (<5). Might be an error. Clamping to 5.0.")
        snr_est = 5.0
    elif snr_est > 100.0:
        print("  ! Very high SNR detected (>100). Possible synthetic data. Clamping to 100.0.")
        snr_est = 100.0
        
    return float(snr_est)


def save_parameter_maps(
    param_maps: dict, 
    affine: np.ndarray, 
    output_dir: str, 
    prefix: str = 'dbsi'
):
    """
    Saves parameter maps as NIfTI files.
    
    Args:
        param_maps: Dictionary with the parameter maps
        affine: Affine matrix from the original volume
        output_dir: Output directory
        prefix: Prefix for the output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    saved_count = 0
    print(f"\n[Utils] Saving {len(param_maps)} maps to: {output_dir}")
    
    for param_name, param_data in param_maps.items():
        try:
            # Ensure data is float32 for saving
            img = nib.Nifti1Image(param_data.astype(np.float32), affine)
            filename = os.path.join(output_dir, f'{prefix}_{param_name}.nii.gz')
            nib.save(img, filename)
            saved_count += 1
        except Exception as e:
            print(f"  ! Error saving {param_name}: {e}")
    
    print(f"  ✓ Saved {saved_count} parameter maps.")