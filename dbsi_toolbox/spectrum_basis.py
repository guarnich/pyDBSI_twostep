# dbsi_toolbox/spectrum_basis.py

import numpy as np
from scipy.optimize import nnls
from .base import BaseDBSI
from .common import DBSIParams

class DBSI_BasisSpectrum(BaseDBSI):  
    """
    DBSI Basis Spectrum solver using standard Scipy NNLS.
    Rimosso ogni riferimento a Numba per garantire stabilità.
    """
    def __init__(self, 
                 iso_diffusivity_range=(0.0, 3.0e-3),
                 n_iso_bases=20,
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
        
        # Stato interno
        self.design_matrix = None
        self.current_bvecs = None
        self.iso_diffusivities = None

    def _build_design_matrix(self, bvals: np.ndarray, bvecs: np.ndarray) -> np.ndarray:
        """
        Costruisce la matrice di design usando NumPy standard.
        """
        n_meas = len(bvals)
        n_aniso = len(bvecs) 
        
        # Griglia diffusività isotrope
        self.iso_diffusivities = np.linspace(
            self.iso_range[0], self.iso_range[1], self.n_iso_bases
        )
        n_iso = len(self.iso_diffusivities)
        
        A = np.zeros((n_meas, n_aniso + n_iso))
        
        # 1. Base Anisotropa (Fibre)
        for j in range(n_aniso):
            fiber_dir = bvecs[j]
            # Calcolo coseno angolo tra gradiente e fibra
            cos_angles = np.dot(bvecs, fiber_dir)
            # Diffusività apparente: D_rad + (D_ax - D_rad) * cos^2(theta)
            D_app = self.radial_diff_basis + (self.axial_diff_basis - self.radial_diff_basis) * (cos_angles**2)
            A[:, j] = np.exp(-bvals * D_app)
            
        # 2. Base Isotropa
        for i, D_iso in enumerate(self.iso_diffusivities):
            A[:, n_aniso + i] = np.exp(-bvals * D_iso)
            
        return A

    def fit_voxel(self, signal: np.ndarray, bvals: np.ndarray) -> DBSIParams:
        """
        Fit singolo voxel usando Scipy.
        """
        if self.design_matrix is None:
            return self._get_empty_params()

        # 1. Normalizzazione e controlli
        if not np.all(np.isfinite(signal)): return self._get_empty_params()
        
        if np.any(bvals < 50):
            S0 = np.mean(signal[bvals < 50])
        else:
            S0 = signal[0]
            
        if S0 <= 1e-6: return self._get_empty_params()
        
        y = signal / S0
        
        # 2. Preparazione matrici per NNLS con regolarizzazione Tikhonov
        # Sistema: ||Ax - y||^2 + lambda*||x||^2
        # Si risolve augumentando A con sqrt(lambda)*I e y con zeri
        
        if self.reg_lambda > 0:
            n_cols = self.design_matrix.shape[1]
            sqrt_lambda = np.sqrt(self.reg_lambda)
            A_aug = np.vstack([self.design_matrix, sqrt_lambda * np.eye(n_cols)])
            y_aug = np.concatenate([y, np.zeros(n_cols)])
        else:
            A_aug = self.design_matrix
            y_aug = y

        # 3. Solver Scipy
        try:
            weights, _ = nnls(A_aug, y_aug)
        except Exception:
            return self._get_empty_params()
        
        # 4. Filtering (Sparsity)
        weights[weights < self.filter_threshold] = 0.0
        
        # 5. Estrazione Metriche
        n_aniso = len(self.current_bvecs)
        
        # Anisotropo
        aniso_weights = weights[:n_aniso]
        f_fiber = np.sum(aniso_weights)
        
        # Direzione principale (corrisponde al peso maggiore nella base anisotropa)
        if f_fiber > 0:
            dom_idx = np.argmax(aniso_weights)
            main_dir = self.current_bvecs[dom_idx]
        else:
            main_dir = np.array([0.0, 0.0, 0.0])
        
        # Isotropo
        iso_weights = weights[n_aniso:]
        
        # Categorie (Wang et al., 2011)
        # Restricted: <= 0.3 um^2/ms
        # Hindered: 0.3 < D <= 2.0 um^2/ms (spesso acqua extracellulare / edema)
        # Free Water: > 2.0 um^2/ms (CSF)
        
        mask_res = self.iso_diffusivities <= 0.3e-3
        mask_hin = (self.iso_diffusivities > 0.3e-3) & (self.iso_diffusivities <= 2.0e-3)
        mask_wat = self.iso_diffusivities > 2.0e-3
        
        f_restricted = np.sum(iso_weights[mask_res])
        f_hindered = np.sum(iso_weights[mask_hin])
        f_water = np.sum(iso_weights[mask_wat])
        
        # Normalizzazione (frazioni relative)
        total = f_fiber + f_restricted + f_hindered + f_water
        if total > 0:
            f_fiber /= total
            f_restricted /= total
            f_hindered /= total
            f_water /= total
        
        # R-Squared (sul segnale originale, non aumentato)
        predicted = self.design_matrix @ weights
        ss_res = np.sum((y - predicted)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return DBSIParams(
            f_restricted=float(f_restricted),
            f_hindered=float(f_hindered),
            f_water=float(f_water),
            f_fiber=float(f_fiber),
            fiber_dir=main_dir,
            axial_diffusivity=self.axial_diff_basis,
            radial_diffusivity=self.radial_diff_basis,
            r_squared=float(r2)
        )