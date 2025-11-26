# dbsi_toolbox/twostep.py
import numpy as np
from .base import BaseDBSI
from .spectrum_basis import DBSI_BasisSpectrum
from .numba_backend import build_design_matrix_numba, fit_volume_numba

class DBSI_TwoStep(BaseDBSI):
    def __init__(self, iso_diffusivity_range=(0.0, 3.0e-3), n_iso_bases=20, reg_lambda=0.01, 
                 filter_threshold=0.01, axial_diff_basis=1.5e-3, radial_diff_basis=0.3e-3):
        self.iso_range = iso_diffusivity_range
        self.n_iso_bases = n_iso_bases
        self.reg_lambda = reg_lambda
        self.axial_diff = axial_diff_basis
        self.radial_diff = radial_diff_basis
        self.spectrum_model = DBSI_BasisSpectrum(iso_diffusivity_range, n_iso_bases, axial_diff_basis, radial_diff_basis, reg_lambda)

    def fit_volume(self, volume, bvals, bvecs, mask=None, show_progress=True, **kwargs):
        X, Y, Z, N = volume.shape
        n_voxels = X * Y * Z
        
        print(f"[DBSI-Fast] Preparing data...")
        data_flat = volume.reshape(n_voxels, N).astype(np.float64)
        mask_flat = mask.flatten().astype(bool) if mask is not None else np.any(data_flat > 0, axis=1)
        flat_bvals = np.array(bvals).flatten().astype(np.float64)
        current_bvecs = bvecs.T.astype(np.float64) if bvecs.shape == (3, N) else bvecs.astype(np.float64)

        print(f"[DBSI-Fast] Building Design Matrix...")
        iso_diffs = np.linspace(self.iso_range[0], self.iso_range[1], self.n_iso_bases)
        idx_res_end = np.sum(iso_diffs <= 0.3e-3)
        idx_hin_end = np.sum(iso_diffs <= 2.0e-3)
        
        design_matrix = build_design_matrix_numba(flat_bvals, current_bvecs, iso_diffs, self.axial_diff, self.radial_diff)
        self.spectrum_model.design_matrix = design_matrix 

        print(f"[DBSI-Fast] Fitting {np.sum(mask_flat)} voxels (Parallel)...")
        raw = fit_volume_numba(data_flat, flat_bvals, design_matrix, self.reg_lambda, mask_flat, len(current_bvecs), idx_res_end, idx_hin_end)
        
        print(f"[DBSI-Fast] Done.")
        return {
            'fiber_fraction': raw[:, 0].reshape(X, Y, Z),
            'restricted_fraction': raw[:, 1].reshape(X, Y, Z),
            'hindered_fraction': raw[:, 2].reshape(X, Y, Z),
            'water_fraction': raw[:, 3].reshape(X, Y, Z),
            'r_squared': raw[:, 4].reshape(X, Y, Z),
            'axial_diffusivity': np.full((X, Y, Z), self.axial_diff),
            'radial_diffusivity': np.full((X, Y, Z), self.radial_diff),
        }