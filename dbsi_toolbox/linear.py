# dbsi_toolbox/linear.py

import numpy as np
from scipy.optimize import nnls
from typing import Tuple
from .base import BaseDBSI
from .common import DBSIParams

class DBSI_Linear(BaseDBSI):
    """
    DBSI implementation using Linear Basis Spectrum and NNLS.
    Best for speed and stability. Robust against solver failures.
    """
    def __init__(self, 
                 iso_diffusivity_range: Tuple[float, float] = (0.0, 3.0e-3),
                 n_iso_bases: int = 20,
                 axial_diff_basis: float = 1.5e-3,
                 radial_diff_basis: float = 0.3e-3):
        
        self.iso_range = iso_diffusivity_range
        self.n_iso_bases = n_iso_bases
        self.axial_diff_basis = axial_diff_basis
        self.radial_diff_basis = radial_diff_basis
        
        # Internal state
        self.design_matrix = None
        self.iso_diffusivities = None

    def fit_volume(self, volume, bvals, bvecs, **kwargs):
        """
        Override to build the design matrix once before iteration.
        """
        flat_bvals = np.array(bvals).flatten()
        N = len(flat_bvals)
        
        if bvecs.shape == (3, N):
            current_bvecs = bvecs.T
        else:
            current_bvecs = bvecs
            
        print("Pre-calculating Linear Design Matrix...", end="")
        self.design_matrix = self._build_design_matrix(flat_bvals, current_bvecs)
        print(" Done.")
        
        return super().fit_volume(volume, bvals, bvecs, **kwargs)

    def _build_design_matrix(self, bvals: np.ndarray, bvecs: np.ndarray) -> np.ndarray:
        n_meas = len(bvals)
        n_aniso = len(bvecs) 
        
        # Generate Isotropic Spectrum
        self.iso_diffusivities = np.linspace(
            self.iso_range[0], self.iso_range[1], self.n_iso_bases
        )
        n_iso = len(self.iso_diffusivities)
        
        A = np.zeros((n_meas, n_aniso + n_iso))
        
        # 1. Anisotropic Basis (Fibers)
        for j in range(n_aniso):
            fiber_dir = bvecs[j]
            norm = np.linalg.norm(fiber_dir)
            if norm > 0: fiber_dir /= norm
                
            cos_angles = np.dot(bvecs, fiber_dir)
            D_app = self.radial_diff_basis + (self.axial_diff_basis - self.radial_diff_basis) * (cos_angles**2)
            A[:, j] = np.exp(-bvals * D_app)
            
        # 2. Isotropic Basis (Spectrum)
        for i, D_iso in enumerate(self.iso_diffusivities):
            A[:, n_aniso + i] = np.exp(-bvals * D_iso)
            
        return A

    def fit_voxel(self, signal: np.ndarray, bvals: np.ndarray) -> DBSIParams:
        if self.design_matrix is None:
            raise RuntimeError("Design matrix not built.")

        # --- ROBUSTNESS CHECK 1: Check for NaNs/Infs ---
        if not np.all(np.isfinite(signal)):
             return self._get_empty_params()

        # Signal Normalization
        if np.any(bvals < 50):
            S0 = np.mean(signal[bvals < 50])
        else:
            S0 = signal[0]
            
        # --- ROBUSTNESS CHECK 2: Check for bad S0 ---
        if S0 <= 1e-6:
            return self._get_empty_params()
            
        y = signal / S0
        
        # --- ROBUSTNESS CHECK 3: Solver Error Handling ---
        try:
            # Solve argmin_x || Ax - y ||_2 subject to x >= 0
            weights, _ = nnls(self.design_matrix, y)
        except RuntimeError:
            # This catches "Maximum number of iterations reached"
            # We return an empty result for this bad voxel
            return self._get_empty_params()
        except Exception:
            # Catches any other unexpected solver error
            return self._get_empty_params()
        
        # Extract Metrics
        n_aniso = len(self.current_bvecs)
        
        # Fiber fractions
        aniso_weights = weights[:n_aniso]
        f_fiber = np.sum(aniso_weights)
        
        # Fiber direction
        if f_fiber > 0:
            dom_idx = np.argmax(aniso_weights)
            main_dir = self.current_bvecs[dom_idx]
        else:
            main_dir = np.array([0.0, 0.0, 0.0])
        
        # Isotropic spectrum aggregation
        iso_weights = weights[n_aniso:]
        
        # Thresholds in mm^2/s (0.3e-3 = 0.3 um^2/ms)
        mask_res = self.iso_diffusivities <= 0.3e-3
        f_restricted = np.sum(iso_weights[mask_res])
        
        mask_hin = (self.iso_diffusivities > 0.3e-3) & (self.iso_diffusivities <= 2.0e-3)
        f_hindered = np.sum(iso_weights[mask_hin])
        
        mask_wat = self.iso_diffusivities > 2.0e-3
        f_water = np.sum(iso_weights[mask_wat])
        
        # R-Squared calculation
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