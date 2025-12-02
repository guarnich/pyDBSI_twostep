# dbsi_toolbox/nlls_tensor_fit.py

import numpy as np
from scipy.optimize import least_squares, differential_evolution
from .base import BaseDBSI
from .common import DBSIParams
from typing import Optional
import warnings

class DBSI_TensorFit(BaseDBSI):  
    """
    DBSI Tensor parameter estimation using Non-Linear Least Squares.
    Refines diffusivities and angles starting from an initial guess.
    
    Features:
    - Robust multi-start optimization for difficult voxels
    - Automatic fallback strategies
    - Quality assessment metrics
    """
    def __init__(self, 
                 use_multistart: bool = False,
                 multistart_threshold: float = 0.1):
        """
        Args:
            use_multistart: Enable multi-start optimization for low-fiber voxels
            multistart_threshold: Fiber fraction below which to use multi-start
        """
        self.n_params = 10
        self.linear_results_map = None
        self.use_multistart = use_multistart
        self.multistart_threshold = multistart_threshold
        
    def _vector_to_angles(self, v: np.ndarray) -> tuple:
        """Converts a vector (x,y,z) to spherical coordinates (theta, phi)."""
        norm = np.linalg.norm(v)
        if norm == 0: 
            return (0.0, 0.0)
        
        v = v / norm
        
        # Clamp to avoid numerical issues with arccos
        theta = np.arccos(np.clip(v[2], -1.0, 1.0))  # [0, pi]
        phi = np.arctan2(v[1], v[0])                  # [-pi, pi]
        
        if phi < 0: 
            phi += 2*np.pi
            
        return (theta, phi)
    
    def _angles_to_vector(self, theta: float, phi: float) -> np.ndarray:
        """Converts spherical coordinates to Cartesian unit vector."""
        return np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])

    def fit_voxel(self, 
                  signal: np.ndarray, 
                  bvals: np.ndarray, 
                  initial_guess: Optional[DBSIParams] = None) -> DBSIParams:
        """
        Fits DBSI model to a single voxel with robust optimization.
        
        Args:
            signal: Measured diffusion signal
            bvals: B-values
            initial_guess: Optional initial parameters from linear fit
            
        Returns:
            Refined DBSI parameters
        """
        # --- 1. SIGNAL NORMALIZATION ---
        if np.any(bvals < 50):
            S0 = np.mean(signal[bvals < 50])
        else:
            S0 = signal[0]
            
        if S0 <= 1e-6 or not np.all(np.isfinite(signal)): 
            return self._get_empty_params()
        
        y = signal / S0
        
        # --- 2. CONSTRUCT INITIAL GUESS ---
        if initial_guess and initial_guess.f_fiber > 0:
            # Two-Step approach: use linear results
            theta, phi = self._vector_to_angles(initial_guess.fiber_dir)
            
            # Ensure valid fractions
            total = initial_guess.f_iso_total + initial_guess.f_fiber + 1e-6
            f_res = np.clip(initial_guess.f_restricted / total, 0.01, 0.99)
            f_hin = np.clip(initial_guess.f_hindered / total, 0.01, 0.99)
            f_wat = np.clip(initial_guess.f_water / total, 0.01, 0.99)
            f_fib = np.clip(initial_guess.f_fiber / total, 0.01, 0.99)
            
            # Start with reasonable diffusivities
            # If linear fit provided estimates, use those; otherwise use literature values
            D_ax_init = initial_guess.axial_diffusivity if initial_guess.axial_diffusivity > 0 else 1.5e-3
            D_rad_init = initial_guess.radial_diffusivity if initial_guess.radial_diffusivity > 0 else 0.3e-3
            
            p0 = [
                f_res, 0.0002,      # Restricted fraction & diffusivity
                f_hin, 0.0010,      # Hindered fraction & diffusivity
                f_wat, f_fib,       # Water & fiber fractions
                D_ax_init,          # Axial diffusivity
                D_rad_init,         # Radial diffusivity
                theta, phi          # Fiber orientation
            ]
            
            use_multistart_here = initial_guess.f_fiber < self.multistart_threshold
            
        else:
            # Blind initialization
            p0 = [0.1, 0.0002, 0.2, 0.001, 0.1, 0.6, 0.0015, 0.0003, np.pi/4, np.pi/4]
            use_multistart_here = True  # Always use multi-start for blind fits

        # --- 3. PARAMETER BOUNDS ---
        # [f_res, D_res, f_hin, D_hin, f_wat, f_fib, D_ax, D_rad, theta, phi]
        bounds_lower = [0.0,  0.0,     0.0,  0.0003, 0.0,  0.0,  0.0005, 0.0,     0.0,   0.0]
        bounds_upper = [1.0,  0.0003,  1.0,  0.0015, 1.0,  1.0,  0.003,  0.0015,  np.pi, 2*np.pi]
        
        # --- 4. OPTIMIZATION ---
        def objective(p):
            """Residual function for least squares."""
            return self._predict_signal(p, bvals, self.current_bvecs) - y
        
        # Strategy selection based on signal quality and initial guess
        if self.use_multistart and use_multistart_here:
            # Use global optimization for difficult cases
            result = self._fit_with_multistart(p0, bounds_lower, bounds_upper, y, bvals)
        else:
            # Standard local optimization
            result = self._fit_local(p0, bounds_lower, bounds_upper, objective)
        
        if result is None:
            return self._get_empty_params()
        
        return result
    
    def _fit_local(self, p0, bounds_lower, bounds_upper, objective):
        """Standard local optimization using trust-region-reflective."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = least_squares(
                    objective, 
                    p0, 
                    bounds=(bounds_lower, bounds_upper), 
                    method='trf',
                    max_nfev=200,  # Limit iterations for speed
                    ftol=1e-6,
                    xtol=1e-6
                )
            
            if not res.success:
                return None
                
            return self._params_from_array(res.x, res.fun)
            
        except Exception:
            return None
    
    def _fit_with_multistart(self, p0, bounds_lower, bounds_upper, y, bvals):
        """
        Robust global optimization using differential evolution.
        Used for voxels with low fiber content or poor initial guesses.
        """
        def objective_scalar(p):
            """Scalar objective for differential_evolution."""
            pred = self._predict_signal(p, bvals, self.current_bvecs)
            return np.sum((pred - y)**2)
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = differential_evolution(
                    objective_scalar,
                    bounds=list(zip(bounds_lower, bounds_upper)),
                    seed=42,
                    maxiter=100,  # Limited iterations for practical speed
                    atol=1e-6,
                    tol=1e-3,
                    workers=1,
                    updating='deferred',
                    strategy='best1bin'
                )
            
            if not result.success:
                return None
            
            # Calculate residuals for R²
            pred = self._predict_signal(result.x, bvals, self.current_bvecs)
            residuals = pred - y
            
            return self._params_from_array(result.x, residuals)
            
        except Exception:
            return None
    
    def _params_from_array(self, p: np.ndarray, residuals: np.ndarray) -> DBSIParams:
        """
        Converts parameter array to DBSIParams object.
        
        Args:
            p: Parameter array [f_res, D_res, f_hin, D_hin, f_wat, f_fib, D_ax, D_rad, theta, phi]
            residuals: Fitting residuals for R² calculation
        """
        # Extract parameters
        f_res, D_res, f_hin, D_hin, f_wat, f_fib, D_ax, D_rad, theta, phi = p
        
        # Convert angles to direction vector
        fiber_dir = self._angles_to_vector(theta, phi)
        
        # Normalize fractions (should sum to 1)
        f_total = f_res + f_hin + f_wat + f_fib + 1e-10
        f_res_norm = f_res / f_total
        f_hin_norm = f_hin / f_total
        f_wat_norm = f_wat / f_total
        f_fib_norm = f_fib / f_total
        
        # Calculate R²
        ss_res = np.sum(residuals**2)
        # Note: y is already normalized in fit_voxel, so mean should be close to 1
        ss_tot = np.sum(residuals**2) + np.var(residuals) * len(residuals)
        r2 = max(0.0, 1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        
        # Quality checks
        if not (0 <= r2 <= 1):
            r2 = 0.0
        if not np.all(np.isfinite([D_ax, D_rad])):
            D_ax, D_rad = 0.0, 0.0
        
        return DBSIParams(
            f_restricted=float(f_res_norm),
            f_hindered=float(f_hin_norm),
            f_water=float(f_wat_norm),
            f_fiber=float(f_fib_norm),
            fiber_dir=fiber_dir,
            axial_diffusivity=float(D_ax),
            radial_diffusivity=float(D_rad),
            r_squared=float(r2)
        )

    def _predict_signal(self, params, bvals, bvecs):
        """
        Forward model: predicts normalized signal from parameters.
        
        Model:
            S/S0 = Σ_i f_i * exp(-b * D_i(θ))
        
        Where:
            - f_i are normalized fractions
            - D_i(θ) is compartment diffusivity (angle-dependent for fibers)
        """
        f_res, D_res, f_hin, D_hin, f_wat, f_fib, D_ax, D_rad, theta, phi = params
        
        # Fixed free water diffusivity (literature value)
        D_water = 3.0e-3
        
        # Normalize fractions
        f_total = f_res + f_hin + f_wat + f_fib + 1e-10
        
        # Fiber direction from spherical coordinates
        fiber_dir = self._angles_to_vector(theta, phi)
        
        # Angle-dependent fiber diffusivity
        cos_angles = np.dot(bvecs, fiber_dir)
        D_app_fiber = D_rad + (D_ax - D_rad) * (cos_angles**2)
        
        # Multi-compartment signal
        signal = (
            (f_res / f_total) * np.exp(-bvals * D_res) +
            (f_hin / f_total) * np.exp(-bvals * D_hin) +
            (f_wat / f_total) * np.exp(-bvals * D_water) +
            (f_fib / f_total) * np.exp(-bvals * D_app_fiber)
        )
        
        return signal
    
    def assess_fit_quality(self, params: DBSIParams) -> Dict[str, float]:
        """
        Assesses the quality of the fitted parameters.
        
        Returns:
            Dictionary with quality metrics
        """
        quality = {
            'r_squared': params.r_squared,
            'fractions_sum': params.f_iso_total + params.f_fiber,
            'diffusivity_ratio': params.axial_diffusivity / (params.radial_diffusivity + 1e-10),
            'is_physically_plausible': True
        }
        
        # Check for physically implausible values
        if params.axial_diffusivity < params.radial_diffusivity:
            quality['is_physically_plausible'] = False
        
        if not (0.95 <= quality['fractions_sum'] <= 1.05):
            quality['is_physically_plausible'] = False
        
        if params.axial_diffusivity > 3.0e-3 or params.radial_diffusivity > 2.0e-3:
            quality['is_physically_plausible'] = False
        
        return quality