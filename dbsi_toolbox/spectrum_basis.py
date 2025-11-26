# dbsi_toolbox/spectrum_basis.py
import numpy as np
from .base import BaseDBSI
from .common import DBSIParams
from .numba_backend import build_design_matrix_numba, fit_volume_numba, coordinate_descent_nnls

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
        self.current_bvecs = None

    def _build_design_matrix(self, bvals, bvecs):
        """Metodo helper per costruire la matrice (usato anche dalla CLI/Jupyter)."""
        flat_bvals = np.array(bvals).flatten().astype(np.float64)
        
        # Standardizzazione bvecs a (N, 3)
        if bvecs.shape == (3, len(flat_bvals)):
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

    def fit_voxel(self, signal, bvals):
        """
        Fit singolo voxel (usato SOLO durante la calibrazione).
        Usa Numba internamente per velocità anche sul singolo voxel.
        """
        if self.design_matrix is None:
            return self._get_empty_params()
            
        s0 = np.mean(signal[bvals < 50]) if np.any(bvals < 50) else signal[0]
        if s0 <= 1e-6: return self._get_empty_params()
        
        y = (signal / s0).astype(np.float64)
        
        # Setup matrici per NNLS
        A = self.design_matrix
        AtA = np.dot(A.T, A)
        if self.reg_lambda > 0:
            # Aggiunta regolarizzazione sulla diagonale
            for i in range(len(AtA)):
                AtA[i, i] += self.reg_lambda
                
        Aty = np.dot(A.T, y)
        
        # Solve
        weights = coordinate_descent_nnls(AtA, Aty)
        
        # Estrazione metriche (semplificata per calibrazione)
        n_aniso = len(self.current_bvecs) if self.current_bvecs is not None else 0
        iso_diffs = np.linspace(self.iso_range[0], self.iso_range[1], self.n_iso_bases)
        
        f_fiber = np.sum(weights[:n_aniso])
        iso_w = weights[n_aniso:]
        
        f_res = np.sum(iso_w[iso_diffs <= 0.3e-3])
        f_hin = np.sum(iso_w[(iso_diffs > 0.3e-3) & (iso_diffs <= 2.0e-3)])
        f_wat = np.sum(iso_w[iso_diffs > 2.0e-3])
        
        return DBSIParams(
            f_restricted=f_res, f_hindered=f_hin, f_water=f_wat, f_fiber=f_fiber,
            fiber_dir=np.zeros(3), axial_diffusivity=0, radial_diffusivity=0, r_squared=0
        )

    def fit_volume(self, volume, bvals, bvecs, mask=None, **kwargs):
        """Non usata direttamente da TwoStep, ma utile se si usa solo Spectrum."""
        pass 