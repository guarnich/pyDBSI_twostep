# dbsi_toolbox/twostep.py
import numpy as np
from .base import BaseDBSI
from .spectrum_basis import DBSI_BasisSpectrum
from .nlls_tensor_fit import DBSI_TensorFit
from .numba_backend import build_design_matrix_numba, fit_volume_numba

class DBSI_TwoStep(BaseDBSI):
    """
    Orchestrator class for DBSI using Numba Acceleration.
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
        
        self.spectrum_model = DBSI_BasisSpectrum(
            iso_diffusivity_range, n_iso_bases, 
            axial_diff_basis, radial_diff_basis, reg_lambda, filter_threshold
        )
        
        self.fitting_model = DBSI_TensorFit()

    def fit_voxel(self, signal, bvals):
        """Wrapper per fit su singolo voxel (usato da calibration.py)"""
        return self.spectrum_model.fit_voxel(signal, bvals)

    def fit_volume(self, volume, bvals, bvecs, mask=None, show_progress=True, **kwargs):
        """
        Esegue il fit volumetrico completo usando l'accelerazione Numba.
        """
        X, Y, Z, N = volume.shape
        n_voxels = X * Y * Z
        
        print(f"[DBSI-Fast] Preparing memory layout...")
        
        # CRITICAL FIX: Forza array contigui in memoria C
        # Se il volume viene da nibabel, spesso è Fortran-contiguous o non contiguo
        data_flat = np.ascontiguousarray(volume.reshape(n_voxels, N), dtype=np.float64)
        
        if mask is not None:
            mask_flat = np.ascontiguousarray(mask.flatten(), dtype=bool)
        else:
            # Auto-mask semplice basato sul segnale medio > 0
            mask_flat = np.any(data_flat > 0, axis=1)
            
        flat_bvals = np.ascontiguousarray(bvals.flatten(), dtype=np.float64)
        
        # Gestione bvecs e trasposizione sicura
        if bvecs.shape == (3, N):
            # Trasponiamo e rendiamo contiguo
            current_bvecs = np.ascontiguousarray(bvecs.T, dtype=np.float64)
        elif bvecs.shape == (N, 3):
            current_bvecs = np.ascontiguousarray(bvecs, dtype=np.float64)
        else:
            raise ValueError(f"Formato bvecs {bvecs.shape} non valido. Atteso (3, N) o (N, 3)")

        print(f"[DBSI-Fast] Building Design Matrix...")
        iso_diffs = np.linspace(self.iso_range[0], self.iso_range[1], self.n_iso_bases)
        
        # Calcolo indici di fine (cumulativi) per le categorie
        # Restricted: [0, 0.3]
        idx_res_end = np.sum(iso_diffs <= 0.3e-3)
        # Hindered: (0.3, 2.0] -> L'indice cumulativo include anche i restricted
        idx_hin_end = np.sum(iso_diffs <= 2.0e-3)
        
        # Costruzione matrice (backend numba)
        design_matrix = build_design_matrix_numba(
            flat_bvals, current_bvecs, iso_diffs, self.axial_diff, self.radial_diff
        )
        # Forza contiguità anche per la matrice di design
        design_matrix = np.ascontiguousarray(design_matrix, dtype=np.float64)
        
        self.spectrum_model.design_matrix = design_matrix 

        print(f"[DBSI-Fast] Fitting {np.sum(mask_flat)} voxels (Parallel)...")
        
        # Chiamata al backend parallelo
        # Assicuriamoci che tutti gli argomenti siano scalari o array contigui
        raw = fit_volume_numba(
            data_flat, 
            flat_bvals, 
            design_matrix, 
            float(self.reg_lambda), 
            mask_flat, 
            int(len(current_bvecs)), 
            int(idx_res_end), 
            int(idx_hin_end)
        )
        
        print(f"[DBSI-Fast] Reconstruction maps...")
        
        # Ricostruzione output 3D
        # raw shape: (n_voxels, 5) -> [fiber, res, hin, wat, r2]
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