# dbsi_toolbox/utils.py

import os
import numpy as np
import nibabel as nib
from typing import Tuple, Optional

# Importa le funzioni di DIPY per la compatibilità
try:
    from dipy.io.image import load_nifti
    from dipy.io import read_bvals_bvecs
    from dipy.core.gradients import gradient_table, GradientTable
except ImportError:
    print("ATTENZIONE: DIPY non trovato. Alcune funzioni di utilità potrebbero non funzionare.")
    print("Installa con: pip install dipy")
    # Definisci tipi fittizi per evitare errori di importazione
    GradientTable = type("GradientTable", (object,), {})


def load_dwi_data_dipy(
    f_nifti: str, 
    f_bval: str, 
    f_bvec: str, 
    f_mask: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, GradientTable, Optional[np.ndarray]]:
    """
    Carica dati DWI, bvals, bvecs e una maschera opzionale usando DIPY.
    
    Args:
        f_nifti: Percorso al file NIfTI 4D (.nii o .nii.gz)
        f_bval: Percorso al file .bval
        f_bvec: Percorso al file .bvec
        f_mask: Percorso opzionale al file NIfTI della maschera 3D
        
    Returns:
        Tupla contenente:
        - data (np.ndarray): Dati DWI 4D
        - affine (np.ndarray): Matrice affine
        - gtab (GradientTable): Oggetto gradient table di DIPY
        - mask (np.ndarray | None): Maschera 3D booleana o None
    """
    print(f"[Utils] Caricamento dati da: {f_nifti}")
    data, affine = load_nifti(f_nifti)
    
    print(f"[Utils] Caricamento bvals/bvecs da: {f_bval}, {f_bvec}")
    bvals, bvecs = read_bvals_bvecs(f_bval, f_bvec)
    gtab = gradient_table(bvals, bvecs)
    
    print(f"  ✓ Volume: {data.shape}, Bvals: {len(gtab.bvals)}, Bvecs: {gtab.bvecs.shape}")
    print(f"  ✓ N. volumi b=0: {np.sum(gtab.b0s_mask)}")
    
    mask_data = None
    if f_mask:
        print(f"[Utils] Caricamento maschera da: {f_mask}")
        mask_data, mask_affine = load_nifti(f_mask)
        mask_data = mask_data.astype(bool)
        
        # Validazione
        if mask_data.shape != data.shape[:3]:
            raise ValueError(
                f"La forma della maschera {mask_data.shape} non corrisponde "
                f"alla forma dei dati {data.shape[:3]}"
            )
        print(f"  ✓ Maschera: {mask_data.shape}, Voxel: {np.sum(mask_data):,}")
    
    return data, affine, gtab, mask_data


def save_parameter_maps(
    param_maps: dict, 
    affine: np.ndarray, 
    output_dir: str, 
    prefix: str = 'dbsi'
):
    """
    Salva le mappe parametriche come file NIfTI
    (Funzione dal tuo script MODELLO FULL)
    
    Args:
        param_maps: Dizionario con le mappe parametriche
        affine: Matrice affine dal volume originale
        output_dir: Directory di output
        prefix: Prefisso per i file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    saved_count = 0
    print(f"\n[Utils] Salvataggio di {len(param_maps)} mappe in: {output_dir}")
    
    for param_name, param_data in param_maps.items():
        try:
            # Assicura che i dati siano float32 per il salvataggio
            img = nib.Nifti1Image(param_data.astype(np.float32), affine)
            filename = os.path.join(output_dir, f'{prefix}_{param_name}.nii.gz')
            nib.save(img, filename)
            saved_count += 1
        except Exception as e:
            print(f"  ! Errore nel salvataggio di {param_name}: {e}")
    
    print(f"  ✓ Salvate {saved_count} mappe parametriche.")