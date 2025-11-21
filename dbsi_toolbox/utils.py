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
    
    # FIX: Passare 'bvecs' come keyword argument per evitare il Warning
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
    
    return data, affine, gtab, mask_data

def estimate_snr(
    data: np.ndarray, 
    gtab: 'GradientTable', #type: ignore
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Stima l'SNR (Signal-to-Noise Ratio) usando le immagini b=0.
    
    Strategia:
    1. Se ci sono >= 3 volumi b=0: Usa il metodo 'temporale' (Media/Std voxel-wise).
    2. Se ci sono < 3 volumi b=0: Usa il metodo 'spaziale' (Segnale ROI / Rumore Background).
    
    Args:
        data: Volume 4D DWI (X, Y, Z, N)
        gtab: GradientTable di DIPY
        mask: Maschera binaria 3D del cervello (opzionale, ma raccomandata)
        
    Returns:
        float: Stima dell'SNR.
    """
    print("\n[Utils] Stima automatica SNR in corso...")
    
    # 1. Estrai i volumi b=0
    b0_mask = gtab.b0s_mask
    b0_data = data[..., b0_mask]
    n_b0 = b0_data.shape[-1]
    
    if n_b0 == 0:
        print("  ! ATTENZIONE: Nessun volume b=0 trovato. Ritorno SNR default = 30.0")
        return 30.0

    # 2. Gestione Maschera (se non fornita, creane una semplice basata sull'intensità)
    if mask is None:
        print("  ! Maschera non fornita. Calcolo maschera base (Otsu thresholding)...")
        from dipy.segment.mask import median_otsu
        # Usa il primo b0 per creare la maschera
        _, mask = median_otsu(b0_data[..., 0], median_radius=2, numpass=1)

    # Assicurati che la maschera sia booleana
    mask = mask.astype(bool)
    if np.sum(mask) == 0:
        print("  ! Maschera vuota. Impossibile calcolare SNR. Ritorno default = 30.0")
        return 30.0

    snr_est = 0.0

    # --- METODO 1: SNR Temporale (se abbiamo abbastanza b0) ---
    if n_b0 >= 3:
        print(f"  ✓ Metodo: Temporale (basato su {n_b0} volumi b=0)")
        # Calcola media e std lungo la dimensione temporale (4a dimensione)
        mean_b0 = np.mean(b0_data, axis=-1)
        std_b0 = np.std(b0_data, axis=-1)
        
        # Evita divisioni per zero
        std_b0[std_b0 == 0] = 1e-10
        
        # SNR voxel-wise
        snr_map = mean_b0 / std_b0
        
        # Prendi la mediana dell'SNR solo all'interno del cervello
        snr_est = np.median(snr_map[mask])
        print(f"  ✓ SNR calcolato (Mediana voxel-wise): {snr_est:.2f}")

    # --- METODO 2: SNR Spaziale (Segnale/Background) ---
    else:
        print(f"  ✓ Metodo: Spaziale (pochi b0 disponibili: {n_b0})")
        # Usa la media di tutti i b0 disponibili per ridurre il rumore visuale
        mean_b0 = np.mean(b0_data, axis=-1)
        
        # Segnale: Media intensità dentro la maschera
        signal_mean = np.mean(mean_b0[mask])
        
        # Rumore: Deviazione standard fuori dalla maschera (Background)
        # Per sicurezza, prendiamo solo gli angoli o i bordi estremi per evitare ghosting
        # Ma per semplicità qui usiamo l'inverso della maschera
        background_mask = ~mask
        
        # Rimuoviamo eventuali artefatti zero/NaN
        noise_data = mean_b0[background_mask]
        noise_data = noise_data[noise_data > 0] 
        
        if len(noise_data) == 0:
             print("  ! Impossibile trovare rumore di fondo valido. Ritorno default = 30.0")
             return 30.0
             
        noise_std = np.std(noise_data)
        
        # Correzione per rumore Rician/Rayleigh nelle immagini magnitudo (background)
        # SD_reale = SD_background / 0.655
        noise_std_corrected = noise_std / 0.655
        
        snr_est = signal_mean / noise_std_corrected
        print(f"  ✓ Segnale medio: {signal_mean:.2f}, Rumore std (corr): {noise_std_corrected:.2f}")
        print(f"  ✓ SNR calcolato: {snr_est:.2f}")

    # Limiti di sicurezza (Sanity Check)
    if snr_est < 5.0:
        print("  ! SNR molto basso rilevato (<5). Potrebbe essere un errore. Clamp a 5.0.")
        snr_est = 5.0
    elif snr_est > 100.0:
        print("  ! SNR molto alto rilevato (>100). Possibile maschera errata o dati sintetici. Clamp a 100.0.")
        snr_est = 100.0
        
    return float(snr_est)


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