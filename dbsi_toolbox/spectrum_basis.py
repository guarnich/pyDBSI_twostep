# dbsi_toolbox/spectrum_basis.py

import numpy as np
from scipy.optimize import nnls
from .base import BaseDBSI
from .common import DBSIParams

class DBSI_BasisSpectrum(BaseDBSI):  
    """
    DBSI Basis Spectrum solver using standard Scipy NNLS.
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
        
        # Internal state
        self.design_matrix = None
        self.current_bvecs = None
        self.iso_diffusivities = None

    def _build_design_matrix(self, bvals: np.ndarray, bvecs: np.ndarray) -> np.ndarray:
        """
        Constructs the DBSI design matrix (A) using standard NumPy.
        Rows: Measurements (b-values).
        Cols: Basis Functions (Anisotropic fibers + Isotropic spectrum).
        """
        n_meas = len(bvals)
        n_aniso = len(bvecs) 
        
        # Create grid of isotropic diffusivities
        self.iso_diffusivities = np.linspace(
            self.iso_range[0], self.iso_range[1], self.n_iso_bases
        )
        n_iso = len(self.iso_diffusivities)
        
        A = np.zeros((n_meas, n_aniso + n_iso))
        
        # 1. Anisotropic Basis (Fibers)
        # We assume fixed axial/radial diffusivities for the basis set
        for j in range(n_aniso):
            fiber_dir = bvecs[j]
            # Calculate cosine of angle between gradient and fiber direction
            cos_angles = np.dot(bvecs, fiber_dir)
            
            # Apparent diffusivity: D_rad + (D_ax - D_rad) * cos^2(theta)
            D_app = self.radial_diff_basis + (self.axial_diff_basis - self.radial_diff_basis) * (cos_angles**2)
            A[:, j] = np.exp(-bvals * D_app)
            
        # 2. Isotropic Basis (Spectrum)
        for i, D_iso in enumerate(self.iso_diffusivities):
            A[:, n_aniso + i] = np.exp(-bvals * D_iso)
            
        return A

    def fit_voxel(self, signal: np.ndarray, bvals: np.ndarray) -> DBSIParams:
        """
        Fits a single voxel using Scipy NNLS with Tikhonov regularization.
        """
        if self.design_matrix is None:
            return self._get_empty_params()

        # 1. Normalization and Safety Checks
        if not np.all(np.isfinite(signal)): return self._get_empty_params()
        
        # Normalize by b0 (assuming b < 50 is effectively b0)
        if np.any(bvals < 50):
            S0 = np.mean(signal[bvals < 50])
        else:
            S0 = signal[0]
            
        if S0 <= 1e-6: return self._get_empty_params()
        
        y = signal / S0
        
        # 2. Prepare matrices for NNLS with Tikhonov Regularization
        # Objective: minimize ||Ax - y||^2 + lambda * ||x||^2
        # This is solved by augmenting A with sqrt(lambda)*I and y with zeros.
        
        if self.reg_lambda > 0:
            n_cols = self.design_matrix.shape[1]
            sqrt_lambda = np.sqrt(self.reg_lambda)
            A_aug = np.vstack([self.design_matrix, sqrt_lambda * np.eye(n_cols)])
            y_aug = np.concatenate([y, np.zeros(n_cols)])
        else:
            A_aug = self.design_matrix
            y_aug = y

        # 3. Scipy NNLS Solver
        try:
            weights, _ = nnls(A_aug, y_aug)
        except Exception:
            return self._get_empty_params()
        
        # 4. Filtering (Sparsity Enforcement)
        # Remove very small weights that are likely noise artifacts
        weights[weights < self.filter_threshold] = 0.0
        
        # 5. Extract Metrics
        n_aniso = len(self.current_bvecs)
        
        # --- Anisotropic Component ---
        aniso_weights = weights[:n_aniso]
        f_fiber = np.sum(aniso_weights)
        
        # Determine dominant direction (corresponds to the highest weight in the anisotropic basis)
        if f_fiber > 0:
            dom_idx = np.argmax(aniso_weights)
            main_dir = self.current_bvecs[dom_idx]
        else:
            main_dir = np.array([0.0, 0.0, 0.0])
        
        # --- Isotropic Component ---
        iso_weights = weights[n_aniso:]
        
        # Categorization based on Wang et al., 2011 / Cross & Song, 2017:
        # Restricted: <= 0.3 um^2/ms (Cellular)
        # Hindered: 0.3 < D <= 2.0 um^2/ms (Extracellular / Edema)
        # Free Water: > 2.0 um^2/ms (CSF)
        
        mask_res = self.iso_diffusivities <= 0.3e-3
        mask_hin = (self.iso_diffusivities > 0.3e-3) & (self.iso_diffusivities <= 2.0e-3)
        mask_wat = self.iso_diffusivities > 2.0e-3
        
        f_restricted = np.sum(iso_weights[mask_res])
        f_hindered = np.sum(iso_weights[mask_hin])
        f_water = np.sum(iso_weights[mask_wat])
        
        # 6. Final Normalization (Relative Fractions)
        total = f_fiber + f_restricted + f_hindered + f_water
        if total > 0:
            f_fiber /= total
            f_restricted /= total
            f_hindered /= total
            f_water /= total
        
        # 7. Calculate R-Squared (on original data, not augmented)
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
            axial_diffusivity=self.axial_diff_basis,   # Fixed basis value in Step 1
            radial_diffusivity=self.radial_diff_basis, # Fixed basis value in Step 1
            r_squared=float(r2)
        )