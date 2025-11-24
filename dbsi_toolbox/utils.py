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
    # Dummy class per evitare crash se dipy manca
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
    
    Priority 2 (Spatial):  Fallback only if 1 b0 exists. Uses Signal(Brain)/Noise(Air).
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
    # Abbassata soglia a 2 volumi come richiesto
    if n_b0 >= 2:
        print(f"  ✓ Method: Temporal (based on {n_b0} b=0 volumes)")
        
        # Media e Deviazione Standard lungo il tempo (4a dimensione)
        mean_b0 = np.mean(b0_data, axis=-1)
        std_b0 = np.std(b0_data, axis=-1)
        
        # Evita divisione per zero
        std_b0[std_b0 == 0] = 1e-10
        
        # Mappa SNR Voxel-wise
        snr_map = mean_b0 / std_b0
        
        # Estrai la mediana dell'SNR SOLO dentro la maschera del cervello
        if np.sum(mask) > 0:
            snr_est = np.median(snr_map[mask])
            print(f"  ✓ Median Temporal SNR (Brain): {snr_est:.2f}")
        else:
            print("  ! Mask is empty. Defaulting to 30.0")
            snr_est = 30.0

    # --- METHOD 2: SPATIAL (Fallback for single b0) ---
    else:
        print(f"  ✓ Method: Spatial (Fallback for single b=0)")
        # Se n_b0 = 1, b0_data ha shape (X,Y,Z,1), prendiamo la slice
        mean_b0 = b0_data[..., 0]
        
        # A. SEGNALE: Mediana dentro la Brain Mask (Input)
        if np.sum(mask) == 0:
            return 30.0
        signal_val = np.median(mean_b0[mask])
        
        # B. RUMORE: Background Automatico (Otsu)
        # Usa Otsu per trovare tutto ciò che non è "Segnale MRI" (Testa/Scalpo)
        # Questo è più sicuro di usare ~mask perché esclude lo scalpo automaticamente
        otsu_thresh, _ = median_otsu(mean_b0, median_radius=2, numpass=1)
        
        # Definisci rumore tutto ciò che è molto sotto la soglia di segnale (es. 50%)
        noise_thresh = otsu_thresh * 0.5
        background_mask = mean_b0 < noise_thresh
        
        # Calcola deviazione standard nel background
        noise_vals = mean_b0[background_mask]
        
        if len(noise_vals) > 100:
            noise_std = np.std(noise_vals)
            # Correzione Rician per il background
            noise_corr = noise_std / 0.655
            
            if noise_corr > 0:
                snr_est = signal_val / noise_corr
                print(f"  ✓ Spatial SNR: {snr_est:.2f} (Sig: {signal_val:.1f}, Noise: {noise_corr:.2f})")
            else:
                snr_est = 30.0
        else:
            print("  ! Insufficient background voxels. Defaulting to 30.0")
            snr_est = 30.0

    # Safety Clamping (per evitare valori assurdi che rompono la calibrazione)
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
            # Salva come float32 per compatibilità
            nib.save(nib.Nifti1Image(v.astype(np.float32), affine), 
                     os.path.join(output_dir, f'{prefix}_{k}.nii.gz'))
        except Exception as e:
            print(f"  ! Error saving {k}: {e}")
            
    print("  ✓ Done.")