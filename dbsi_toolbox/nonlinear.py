# dbsi_toolbox/nonlinear.py

import numpy as np
from scipy.optimize import least_squares
from .base import BaseDBSI
from .common import DBSIParams
from typing import Optional

class DBSI_NonLinear(BaseDBSI):
    """
    DBSI implementation using Non-Linear Least Squares (NLLS).
    Can be initialized with results from the Linear model (Two-Step Approach).
    """
    def __init__(self):
        self.n_params = 10
        # Placeholder for the linear results volume if Two-Step is used
        self.linear_results_map = None 
        
    def _vector_to_angles(self, v: np.ndarray) -> tuple:
        """Converts a vector (x,y,z) to (theta, phi)."""
        norm = np.linalg.norm(v)
        if norm == 0: return (0.0, 0.0)
        v = v / norm
        theta = np.arccos(v[2]) # 0 to pi
        phi = np.arctan2(v[1], v[0]) # -pi to pi
        if phi < 0: phi += 2*np.pi
        return (theta, phi)

    def fit_voxel(self, signal: np.ndarray, bvals: np.ndarray, 
                  initial_guess: Optional[DBSIParams] = None) -> DBSIParams:
        
        # Signal Normalization
        if np.any(bvals < 50):
            S0 = np.mean(signal[bvals < 50])
        else:
            S0 = signal[0]
            
        if S0 <= 1e-6 or not np.all(np.isfinite(signal)): 
            return self._get_empty_params()
        y = signal / S0
        
        # --- 1. DETERMINE INITIAL GUESS (p0) ---
        if initial_guess and initial_guess.f_fiber > 0:
            # Use Linear results ("Two-Step")
            theta, phi = self._vector_to_angles(initial_guess.fiber_dir)
            
            # Ensure valid fractions for p0 (avoid 0 to prevent log errors)
            total = initial_guess.f_iso_total + initial_guess.f_fiber + 1e-6
            f_res = max(0.01, initial_guess.f_restricted / total)
            f_hin = max(0.01, initial_guess.f_hindered / total)
            f_wat = max(0.01, initial_guess.f_water / total)
            f_fib = max(0.01, initial_guess.f_fiber / total)
            
            # [f_res, D_res, f_hin, D_hin, f_wat, f_fib, D_ax, D_rad, theta, phi]
            p0 = [
                f_res, 0.0002, 
                f_hin, 0.0010, 
                f_wat, f_fib, 
                1.5e-3, 0.3e-3, # Start with standard diffusivities
                theta, phi
            ]
        else:
            # Blind guess (if Linear failed or not provided)
            p0 = [0.1, 0.0002, 0.2, 0.001, 0.1, 0.6, 0.0015, 0.0003, np.pi/4, np.pi/4]

        # --- 2. SETUP OPTIMIZATION ---
        bounds_lower = [0.0, 0.0,     0.0, 0.0003, 0.0, 0.0, 0.0005, 0.0,     0.0,   0.0]
        bounds_upper = [1.0, 0.0003,  1.0, 0.0015, 1.0, 1.0, 0.003,  0.0015, np.pi, 2*np.pi]
        
        def objective(p):
            return self._predict_signal(p, bvals, self.current_bvecs) - y
        
        try:
            res = least_squares(objective, p0, bounds=(bounds_lower, bounds_upper), method='trf')
            p = res.x
            
            # Convert angles back to vector
            fiber_dir = np.array([
                np.sin(p[8]) * np.cos(p[9]),
                np.sin(p[8]) * np.sin(p[9]),
                np.cos(p[8])
            ])
            
            # Normalize fractions
            f_total = p[0] + p[2] + p[4] + p[5] + 1e-10
            
            # R-Squared
            ss_res = np.sum(res.fun**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1 - (ss_res/ss_tot) if ss_tot > 0 else 0
            
            return DBSIParams(
                f_restricted=p[0]/f_total,
                f_hindered=p[2]/f_total,
                f_water=p[4]/f_total,
                f_fiber=p[5]/f_total,
                fiber_dir=fiber_dir,
                axial_diffusivity=p[6],
                radial_diffusivity=p[7],
                r_squared=r2
            )
        except Exception:
            return self._get_empty_params()

    def _predict_signal(self, params, bvals, bvecs):
        f_res, D_res, f_hin, D_hin, f_wat, f_fib, D_ax, D_rad, theta, phi = params
        D_water = 3.0e-3
        
        f_total = f_res + f_hin + f_wat + f_fib + 1e-10
        
        fiber_dir = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        
        cos_angles = np.dot(bvecs, fiber_dir)
        D_app_fiber = D_rad + (D_ax - D_rad) * (cos_angles**2)
        
        signal = (
            (f_res / f_total) * np.exp(-bvals * D_res) +
            (f_hin / f_total) * np.exp(-bvals * D_hin) +
            (f_wat / f_total) * np.exp(-bvals * D_water) +
            (f_fib / f_total) * np.exp(-bvals * D_app_fiber)
        )
        return signal