# dbsi_toolbox/utils.py

import os
import numpy as np
import nibabel as nib
from typing import Tuple, Optional

# Import DIPY functions for compatibility
try:
    from dipy.io.image import load_nifti
    from dipy.io import read_bvals_bvecs
    from dipy.core.gradients import gradient_table, GradientTable
except ImportError:
    print("WARNING: DIPY not found. Some utility functions may not work.")
    print("Install with: pip install dipy")
    
    # Create a dummy type to avoid import errors if dipy isn't present
    GradientTable = type("GradientTable", (object,), {})


def load_dwi_data_dipy(
    f_nifti: str, 
    f_bval: str, 
    f_bvec: str, 
    f_mask: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, GradientTable, Optional[np.ndarray]]:
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
    gtab = gradient_table(bvals, bvecs)
    
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
    
    return data, affine, gtab, mask_data


def save_parameter_maps(
    param_maps: dict, 
    affine: np.ndarray, 
    output_dir: str, 
    prefix: str = 'dbsi'
):
    """
    Saves parameter maps as NIfTI files
    (Function from your FULL MODEL script)
    
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