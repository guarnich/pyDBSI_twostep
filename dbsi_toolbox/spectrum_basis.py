# dbsi_toolbox/spectrum_basis.py

import numpy as np
from .base import BaseDBSI
from .common import DBSIParams
from .numba_backend import build_design_matrix_numba, fit_volume_numba

class DBSI_BasisSpectrum(BaseDBSI):
    """
    DBSI Basis Spectrum solver using Numba acceleration.
    """
    def __init__(self, 
                 iso_diffusivity_range=(0.0, 3.0e-3),
                 n_iso_bases=25,
                 axial_diff_basis=1.5e-3,
                 radial_diff_basis=0.3e-3,
                 reg_lambda=0.01,
                 filter_threshold=0.01):
        
        self.iso_range = iso_diffusivity_range
        self.n_iso_bases = n_iso_bases
        self.axial_diff_basis = axial_diff_basis
        self.radial_diff_basis = radial_diff_basis
        self.reg_lambda = reg_lambda
        self.filter_threshold = filter_threshold
        self.design_matrix = None

    def _build_design_matrix(self, bvals, bvecs):
        """
        Helper per costruire la matrice di design (usato per visualizzazione nel notebook).
        """
        flat_bvals = np.array(bvals).flatten().astype(np.float64)
        n_meas = len(flat_bvals)
        
        # Standardize bvecs to (N, 3)
        # Se arrivano come (3, N), li trasponiamo
        if bvecs.shape == (3, n_meas):
            current_bvecs = bvecs.T.astype(np.float64)
        else:
            current_bvecs = bvecs.astype(np.float64)
            
        iso_diffs = np.linspace(self.iso_range[0], self.iso_range[1], self.n_iso_bases)
        
        return build_design_matrix_numba(
            flat_bvals, 
            current_bvecs, 
            iso_diffs, 
            self.axial_diff_basis, 
            self.radial_diff_basis
        )

    def fit_volume(self, volume, bvals, bvecs, mask=None, **kwargs):
        """
        Versione accelerata Numba del fitting volumetrico (standalone).
        """
        X, Y, Z, N = volume.shape
        n_voxels = X * Y * Z
        
        # 1. Flattening dati
        data_flat = volume.reshape(n_voxels, N).astype(np.float64)
        mask_flat = mask.flatten().astype(bool) if mask is not None else np.ones(n_voxels, dtype=bool)
        
        flat_bvals = np.array(bvals).flatten().astype(np.float64)
        if bvecs.shape == (3, N):
            current_bvecs = bvecs.T.astype(np.float64)
        else:
            current_bvecs = bvecs.astype(np.float64)

        # 2. Design Matrix
        if self.design_matrix is None:
            self.design_matrix = self._build_design_matrix(flat_bvals, current_bvecs)
        
        # 3. Indici per frazioni isotrope
        iso_diffs = np.linspace(self.iso_range[0], self.iso_range[1], self.n_iso_bases)
        idx_res_end = np.sum(iso_diffs <= 0.3e-3)
        idx_hin_end = np.sum(iso_diffs <= 2.0e-3)
        n_aniso = len(current_bvecs)

        # 4. Fit
        raw_results = fit_volume_numba(
            data_flat, 
            flat_bvals, 
            self.design_matrix, 
            self.reg_lambda, 
            mask_flat,
            n_aniso,
            idx_res_end,
            idx_hin_end
        )
        
        # 5. Reshape
        maps = {
            'fiber_fraction': raw_results[:, 0].reshape(X, Y, Z),
            'restricted_fraction': raw_results[:, 1].reshape(X, Y, Z),
            'hindered_fraction': raw_results[:, 2].reshape(X, Y, Z),
            'water_fraction': raw_results[:, 3].reshape(X, Y, Z),
            'r_squared': raw_results[:, 4].reshape(X, Y, Z)
        }
        return maps