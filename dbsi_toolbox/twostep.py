# dbsi_toolbox/twostep.py
import numpy as np
import sys
from .base import BaseDBSI
from .spectrum_basis import DBSI_BasisSpectrum
from .nlls_tensor_fit import DBSI_TensorFit  # Importiamo la classe vera
from .numba_backend import build_design_matrix_numba, fit_volume_numba

class DBSI_TwoStep(BaseDBSI):
    """
    Orchestrator class for DBSI.
    Combines Numba-accelerated Basis Spectrum and Tensor Fit logic.
    """
    def __init__(self, 
                 iso_diffusivity_range=(0.0, 3.0e-3),
                 n_iso_bases=20,
                 reg_lambda=0.01,
                 filter_threshold=0.01,
                 axial_diff_basis=1.5e-3,
                 radial_diff_basis=0.3e-3):
        
        self.iso_range = iso_diffusivity_range
        self.n_iso_bases = n_iso_bases
        self.reg_lambda = reg_lambda
        self.axial_diff = axial_diff_basis
        self.radial_diff = radial_diff_basis
        
        # Inizializza i modelli interni reali (Niente Mock)
        self.spectrum_model = DBSI_BasisSpectrum(
            iso_diffusivity_range, n_iso_bases, 
            axial_diff_basis, radial_diff_basis, reg_lambda, filter_threshold
        )
        
        # Manteniamo questo per compatibilità con calibration.py e future estensioni non-lineari
        self.fitting_model = DBSI_TensorFit()

    def fit_voxel(self, signal, bvals):
        """Wrapper per fit su singolo voxel (usato da calibration.py)"""
        # Delega al modello spettrale che ora ha il metodo fit_voxel Numba-aware
        return self.spectrum_model.fit_voxel(signal, bvals)

    def fit_volume(self, volume, bvals, bvecs, mask=None, show_progress=True, **kwargs):
        """
        Esegue il fit volumetrico completo usando l'accelerazione Numba.
        """
        X, Y, Z, N = volume.shape
        n_voxels = X * Y * Z
        
        print(f"[DBSI-Fast] Preparing data structures...")
        data_flat = volume.reshape(n_voxels, N).astype(np.float64)
        
        if mask is not None:
            mask_flat = mask.flatten().astype(bool)
        else:
            mask_flat = np.any(data_flat > 0, axis=1) # Auto-mask semplice
            
        flat_bvals = np.array(bvals).flatten().astype(np.float64)
        
        # Gestione bvecs
        if bvecs.shape == (3, N):
            current_bvecs = bvecs.T.astype(np.float64)
        else:
            current_bvecs = bvecs.astype(np.float64)

        print(f"[DBSI-Fast] Building Design Matrix...")
        iso_diffs = np.linspace(self.iso_range[0], self.iso_range[1], self.n_iso_bases)
        
        # Calcolo indici per aggregazione frazioni
        idx_res_end = np.sum(iso_diffs <= 0.3e-3)
        idx_hin_end = np.sum(iso_diffs <= 2.0e-3)
        
        # Costruiamo e salviamo la matrice (utile anche per debug/plot esterni)
        design_matrix = build_design_matrix_numba(
            flat_bvals, current_bvecs, iso_diffs, self.axial_diff, self.radial_diff
        )
        self.spectrum_model.design_matrix = design_matrix 

        print(f"[DBSI-Fast] Fitting {np.sum(mask_flat)} voxels (Parallel)...")
        
        # Chiamata al backend parallelo
        raw = fit_volume_numba(
            data_flat, 
            flat_bvals, 
            design_matrix, 
            self.reg_lambda, 
            mask_flat, 
            len(current_bvecs), 
            idx_res_end, 
            idx_hin_end
        )
        
        print(f"[DBSI-Fast] Reconstruction maps...")
        
        # Ricostruzione output 3D
        maps = {
            'fiber_fraction': raw[:, 0].reshape(X, Y, Z),
            'restricted_fraction': raw[:, 1].reshape(X, Y, Z),
            'hindered_fraction': raw[:, 2].reshape(X, Y, Z),
            'water_fraction': raw[:, 3].reshape(X, Y, Z),
            'r_squared': raw[:, 4].reshape(X, Y, Z),
            'axial_diffusivity': np.full((X, Y, Z), self.axial_diff),
            'radial_diffusivity': np.full((X, Y, Z), self.radial_diff),
        }
        
        return maps