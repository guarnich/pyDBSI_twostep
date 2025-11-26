# dbsi_toolbox/spectrum_basis.py
import numpy as np
from .base import BaseDBSI
from .common import DBSIParams
from .numba_backend import build_design_matrix_numba, coordinate_descent_nnls

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
        """Metodo helper per costruire la matrice in modo sicuro per Numba."""
        # Contiguità e tipi
        flat_bvals = np.ascontiguousarray(bvals.flatten(), dtype=np.float64)
        
        if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
             # (3, N) -> (N, 3)
            current_bvecs = np.ascontiguousarray(bvecs.T, dtype=np.float64)
        else:
            current_bvecs = np.ascontiguousarray(bvecs, dtype=np.float64)
            
        iso_diffs = np.linspace(self.iso_range[0], self.iso_range[1], self.n_iso_bases)
        
        A = build_design_matrix_numba(
            flat_bvals, 
            current_bvecs, 
            iso_diffs, 
            self.axial_diff_basis, 
            self.radial_diff_basis
        )
        return np.ascontiguousarray(A)

    def fit_voxel(self, signal, bvals):
        """
        Fit singolo voxel (usato SOLO durante la calibrazione).
        """
        if self.design_matrix is None:
            return self._get_empty_params()
            
        # Normalizzazione
        s0 = np.mean(signal[bvals < 50]) if np.any(bvals < 50) else signal[0]
        if s0 <= 1e-6: return self._get_empty_params()
        
        y = np.ascontiguousarray(signal / s0, dtype=np.float64)
        
        # Setup matrici per NNLS
        # Poiché questo metodo viene chiamato ripetutamente in calibration.py,
        # ricalcoliamo AtA qui per evitare di sporcare la matrice globale
        A = self.design_matrix
        AtA = np.dot(A.T, A)
        
        if self.reg_lambda > 0:
            for i in range(len(AtA)):
                AtA[i, i] += self.reg_lambda
                
        Aty = np.dot(A.T, y)
        
        # Solve
        weights = coordinate_descent_nnls(AtA, Aty)
        
        # Estrazione metriche
        n_aniso = self.design_matrix.shape[1] - self.n_iso_bases
        iso_diffs = np.linspace(self.iso_range[0], self.iso_range[1], self.n_iso_bases)
        
        f_fiber = np.sum(weights[:n_aniso])
        iso_w = weights[n_aniso:]
        
        f_res = np.sum(iso_w[iso_diffs <= 0.3e-3])
        f_hin = np.sum(iso_w[(iso_diffs > 0.3e-3) & (iso_diffs <= 2.0e-3)])
        f_wat = np.sum(iso_w[iso_diffs > 2.0e-3])
        
        # Normalizzazione per l'oggetto parametri
        total = f_fiber + f_res + f_hin + f_wat
        if total > 0:
            f_fiber /= total
            f_res /= total
            f_hin /= total
            f_wat /= total
        
        return DBSIParams(
            f_restricted=f_res, f_hindered=f_hin, f_water=f_wat, f_fiber=f_fiber,
            fiber_dir=np.zeros(3), axial_diffusivity=self.axial_diff_basis, 
            radial_diffusivity=self.radial_diff_basis, r_squared=0.0
        )