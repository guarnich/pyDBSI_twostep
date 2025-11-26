# dbsi_toolbox/spectrum_basis.py (Aggiornato)

import numpy as np
from .base import BaseDBSI
from .common import DBSIParams
from .numba_backend import build_design_matrix_numba, fit_volume_numba # Importa il nuovo modulo

class DBSI_BasisSpectrum(BaseDBSI):
    def fit_volume(self, volume, bvals, bvecs, mask=None, **kwargs):
        """
        Versione accelerata con Numba.
        """
        X, Y, Z, N = volume.shape
        n_voxels = X * Y * Z
        
        # 1. Flattening dei dati per Numba
        data_flat = volume.reshape(n_voxels, N).astype(np.float64)
        mask_flat = mask.flatten().astype(bool) if mask is not None else np.ones(n_voxels, dtype=bool)
        
        flat_bvals = np.array(bvals).flatten().astype(np.float64)
        # Assicura bvecs siano (N, 3)
        if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
            current_bvecs = bvecs.T.astype(np.float64)
        else:
            current_bvecs = bvecs.astype(np.float64)

        # 2. Costruzione Matrice A con Numba
        print("Building Design Matrix (Numba)...")
        # Definisci le diffusività isotrope (da self.iso_range e self.n_iso_bases)
        iso_diffs = np.linspace(self.iso_range[0], self.iso_range[1], self.n_iso_bases)
        
        self.design_matrix = build_design_matrix_numba(
            flat_bvals, 
            current_bvecs, 
            iso_diffs, 
            self.axial_diff_basis, 
            self.radial_diff_basis
        )
        
        # 3. Fitting Parallelo con Numba
        print(f"Fitting {np.sum(mask_flat)} voxels (Numba Parallel)...")
        # Nota: fit_volume_numba restituisce una matrice grezza (N_vox, 5)
        # Bisogna passare gli indici per suddividere le frazioni isotrope (omesso per brevità nel codice sopra,
        # ma essenziale nell'implementazione completa).
        raw_results = fit_volume_numba(
            data_flat, 
            flat_bvals, 
            current_bvecs, 
            self.design_matrix, 
            self.reg_lambda, 
            mask_flat
        )
        
        # 4. Reshape e Salvataggio nelle mappe
        maps = {
            'fiber_fraction': raw_results[:, 0].reshape(X, Y, Z),
            # ... altri canali
            'r_squared': raw_results[:, 4].reshape(X, Y, Z)
        }
        
        return maps