# dbsi_toolbox/model.py

import numpy as np
from scipy.optimize import least_squares, differential_evolution
from dataclasses import dataclass
from typing import Tuple, Optional, List
from tqdm import tqdm
import warnings
import sys

# DBSIParams class for structuring results
@dataclass
class DBSIParams:
    """Dataclass for complete DBSI parameters"""
    # Isotropic components
    f_restricted: float      # Restricted fraction (cells, inflammation)
    D_restricted: float      # Restricted diffusivity (~0.0-0.3 μm²/ms)
    f_hindered: float        # Hindered fraction (extracellular space)
    D_hindered: float        # Hindered diffusivity (~0.3-1.5 μm²/ms)
    f_water: float           # Free water fraction (CSF, edema)
    D_water: float           # Water diffusivity (~2.5-3.0 μm²/ms)
    
    # Anisotropic component
    f_fiber: float           # Fiber fraction
    D_axial: float           # Axial diffusivity
    D_radial: float          # Radial diffusivity
    theta: float             # Polar angle
    phi: float               # Azimuthal angle
    
    @property
    def f_iso_total(self) -> float:
        """Total isotropic fraction"""
        return self.f_restricted + self.f_hindered + self.f_water
    
    @property
    def f_aniso_total(self) -> float:
        """Total anisotropic fraction"""
        return self.f_fiber
    
    @property
    def fiber_FA(self) -> float:
        """Fractional Anisotropy of the fiber compartment"""
        mean_D = (self.D_axial + 2*self.D_radial) / 3
        if mean_D < 1e-10: # Avoid division by zero
            return 0.0
        
        # Standard FA formula
        numerator = (self.D_axial - mean_D)**2 + 2*(self.D_radial - mean_D)**2
        denominator = self.D_axial**2 + 2*self.D_radial**2
        if denominator < 1e-10:
            return 0.0
        
        return np.sqrt(0.5 * numerator / denominator)
    
    @property
    def fiber_MD(self) -> float:
        """Mean Diffusivity of the fiber compartment"""
        return (self.D_axial + 2*self.D_radial) / 3
    
    @property
    def overall_FA(self) -> float:
        """Overall weighted FA"""
        total_signal = self.f_fiber + self.f_iso_total
        if total_signal < 1e-10 or self.f_fiber < 1e-10:
            return 0.0
        # FA weighted by the anisotropic fraction
        return self.fiber_FA * (self.f_fiber / total_signal)
    
    @property
    def overall_MD(self) -> float:
        """Overall MD weighted by all compartments"""
        total_f = self.f_restricted + self.f_hindered + self.f_water + self.f_fiber
        if total_f < 1e-10:
            return 0.0
        weighted_D = (self.f_restricted * self.D_restricted + 
                     self.f_hindered * self.D_hindered +
                     self.f_water * self.D_water +
                     self.f_fiber * self.fiber_MD)
        return weighted_D / total_f
    
    @property
    def cellularity_index(self) -> float:
        """Cellularity index (for inflammation/tumors)"""
        iso_total = self.f_iso_total
        if iso_total < 1e-10:
            return 0.0
        return self.f_restricted / iso_total
    
    @property
    def edema_index(self) -> float:
        """Edema index (free water fraction)"""
        iso_total = self.f_iso_total
        if iso_total < 1e-10:
            return 0.0
        return self.f_water / iso_total
    
    @property
    def fiber_density(self) -> float:
        """Normalized fiber density"""
        total_signal = self.f_fiber + self.f_iso_total
        if total_signal < 1e-10:
            return 0.0
        return self.f_fiber / total_signal
    
    @property
    def axial_diffusivity(self) -> float:
        """Axial diffusivity (alias for compatibility)"""
        return self.D_axial
    
    @property
    def radial_diffusivity(self) -> float:
        """Radial diffusivity (alias for compatibility)"""
        return self.D_radial

# The DBSIModel class for fitting
class DBSIModel:
    """Complete DBSI model for multi-compartment fitting"""
    
    def __init__(self):
        """Initialize the complete DBSI model"""
        self.n_params = 10  # 3 iso (f,D) + 1 aniso (f,D_ax,D_rad,theta,phi), D_water fixed
    
    def predict_signal(self, params: np.ndarray, bvals: np.ndarray, 
                      bvecs: np.ndarray, S0: float = 1.0) -> np.ndarray:
        """
        Predicts the DWI signal using the complete DBSI model
        
        Args:
            params: [f_res, D_res, f_hin, D_hin, f_wat, f_fib, D_ax, D_rad, theta, phi]
            bvals: b-values (N,)
            bvecs: Normalized direction vectors (N, 3)
            S0: Signal at b=0
            
        Returns:
            Predicted signal (N,)
        """
        f_res, D_res, f_hin, D_hin, f_wat, f_fib, D_ax, D_rad, theta, phi = params
        
        # Fixed free water diffusivity (typical for CSF)
        D_water = 3.0e-3  # 3.0 μm²/ms
        
        # Normalize fractions
        f_total = f_res + f_hin + f_wat + f_fib
        if f_total > 1e-10:
            f_res_norm = f_res / f_total
            f_hin_norm = f_hin / f_total
            f_wat_norm = f_wat / f_total
            f_fib_norm = f_fib / f_total
        else:
            # Default if sum is zero
            f_res_norm = 0.0
            f_hin_norm = 0.0
            f_wat_norm = 0.0
            f_fib_norm = 0.0
        
        # Main fiber direction
        fiber_dir = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        
        # Anisotropic diffusion tensor (fibers)
        D_tensor = D_rad * np.eye(3) + (D_ax - D_rad) * np.outer(fiber_dir, fiber_dir)
        
        # Signal calculation (vectorized for efficiency)
        
        # Isotropic components
        S_restricted = np.exp(-bvals * D_res)
        S_hindered = np.exp(-bvals * D_hin)
        S_water = np.exp(-bvals * D_water)
        
        # Anisotropic component
        # Calculate (g * D * g) for all bvecs
        g_D_g = np.einsum('ij,ji->i', np.dot(bvecs, D_tensor), bvecs.T)
        S_fiber = np.exp(-bvals * g_D_g)
        
        # Total weighted signal
        signal = S0 * (
            f_res_norm * S_restricted +
            f_hin_norm * S_hindered +
            f_wat_norm * S_water +
            f_fib_norm * S_fiber
        )
        
        return signal
    
    def fit_voxel(self, signal: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray,
                  method: str = 'least_squares') -> Tuple[np.ndarray, dict]:
        """
        Performs fitting for a single voxel
        
        Args:
            signal: Observed DWI signal (N,)
            bvals: b-values (N,)
            bvecs: Direction vectors (N, 3)
            method: 'least_squares' or 'differential_evolution'
            
        Returns:
            Fitted parameters array and fitting info
        """
        # Normalize the signal
        S0 = np.mean(signal[bvals < 50]) if np.any(bvals < 50) else signal[0]
        
        # Check if the voxel has valid signal
        if S0 < 1e-6 or np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
            return np.zeros(self.n_params), {'success': False, 'r_squared': 0.0, 'S0': 0.0}
        
        signal_norm = signal / (S0 + 1e-10)
        
        # Initial parameters and bounds
        # [f_res, D_res, f_hin, D_hin, f_wat, f_fib, D_ax, D_rad, theta, phi]
        params0 = [0.1, 0.0002, 0.2, 0.001, 0.1, 0.6, 0.0015, 0.0003, np.pi/4, np.pi/4]
        bounds_lower = [0.0, 0.0,     0.0, 0.0003, 0.0, 0.0, 0.0005, 0.0,     0.0,   0.0]
        bounds_upper = [1.0, 0.0003,  1.0, 0.0015, 1.0, 1.0, 0.003,  0.0015, np.pi, 2*np.pi]
        
        def objective(p):
            pred = self.predict_signal(p, bvals, bvecs, S0=1.0)
            return pred - signal_norm
        
        # Optimization
        try:
            if method == 'least_squares':
                result = least_squares(
                    objective, 
                    params0,
                    bounds=(bounds_lower, bounds_upper),
                    method='trf',
                    ftol=1e-6,
                    xtol=1e-6,
                    max_nfev=1000
                )
                params_opt = result.x
                success = result.success
                
            elif method == 'differential_evolution':
                result = differential_evolution(
                    lambda p: np.sum(objective(p)**2),
                    bounds=list(zip(bounds_lower, bounds_upper)),
                    seed=42,
                    maxiter=500,
                    polish=True
                )
                params_opt = result.x
                success = result.success
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            # Calculate R²
            signal_pred = self.predict_signal(params_opt, bvals, bvecs, S0)
            ss_res = np.sum((signal - signal_pred)**2)
            ss_tot = np.sum((signal - np.mean(signal))**2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-10))
            
            info = {
                'success': success,
                'r_squared': r_squared,
                'S0': S0
            }
            
            return params_opt, info
            
        except Exception as e:
            warnings.warn(f"Fitting failed: {str(e)}")
            return np.zeros(self.n_params), {'success': False, 'r_squared': 0.0, 'S0': S0}
    
    def fit_volume(self, volume: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray,
                   mask: Optional[np.ndarray] = None, method: str = 'least_squares',
                   show_progress: bool = True) -> dict:
        """
        Performs fitting on a 4D volume
        
        Args:
            volume: 4D DWI Volume (X, Y, Z, N)
            bvals: b-values (N,) or (1, N)
            bvecs: Direction vectors (3, N) or (N, 3)
            mask: 3D Mask (X, Y, Z) - optional
            method: 'least_squares' or 'differential_evolution'
            show_progress: Show progress bar
            
        Returns:
            Dictionary with 3D parameter maps
        """
        # Verify dimensions
        if volume.ndim != 4:
            raise ValueError(f"Volume must be 4D, but has {volume.ndim} dimensions")
        
        X, Y, Z, N = volume.shape
        
        # Handle bvals
        if bvals.ndim == 2:
            bvals = bvals.flatten()
        if len(bvals) != N:
            raise ValueError(f"bvals has {len(bvals)} elements, but volume has {N} volumes")
        
        # Handle bvecs: convert from (3, N) to (N, 3) if necessary
        if bvecs.shape == (3, N):
            bvecs = bvecs.T
        elif bvecs.shape != (N, 3):
            raise ValueError(f"bvecs must be (3, {N}) or ({N}, 3), but shape is {bvecs.shape}")
        
        # Normalize bvecs
        bvecs_norm = bvecs / (np.linalg.norm(bvecs, axis=1, keepdims=True) + 1e-10)
        
        # Create or validate mask
        if mask is None:
            b0_idx = np.argmin(bvals)
            b0_volume = volume[:, :, :, b0_idx]
            threshold = np.percentile(b0_volume[b0_volume > 0], 10)
            mask = b0_volume > threshold
        
        if mask.shape != (X, Y, Z):
            raise ValueError(f"Mask must have dimensions ({X}, {Y}, {Z}), but has {mask.shape}")
        
        # Initialize ALL possible parameter maps
        param_maps = {
            # Specific compartment fractions
            'f_restricted': np.zeros((X, Y, Z)),
            'D_restricted': np.zeros((X, Y, Z)),
            'f_hindered': np.zeros((X, Y, Z)),
            'D_hindered': np.zeros((X, Y, Z)),
            'f_water': np.zeros((X, Y, Z)),
            'f_fiber': np.zeros((X, Y, Z)),
            'D_axial': np.zeros((X, Y, Z)),
            'D_radial': np.zeros((X, Y, Z)),
            
            # Aggregated fractions
            'f_iso': np.zeros((X, Y, Z)),      # Sum of all isotropic fractions
            'f_aniso': np.zeros((X, Y, Z)),    # Anisotropic fraction (alias for f_fiber)
            
            # Fiber orientation
            'theta': np.zeros((X, Y, Z)),
            'phi': np.zeros((X, Y, Z)),
            
            # Anisotropy and diffusivity metrics
            'fiber_FA': np.zeros((X, Y, Z)),         # FA of the fiber compartment
            'fiber_MD': np.zeros((X, Y, Z)),         # MD of the fiber compartment
            'overall_FA': np.zeros((X, Y, Z)),       # Overall weighted FA
            'overall_MD': np.zeros((X, Y, Z)),       # Overall weighted MD
            'FA': np.zeros((X, Y, Z)),               # Alias for overall_FA
            'MD': np.zeros((X, Y, Z)),               # Alias for overall_MD
            'AD': np.zeros((X, Y, Z)),               # Axial Diffusivity (D_axial)
            'RD': np.zeros((X, Y, Z)),               # Radial Diffusivity (D_radial)
            
            # Pathological indices
            'cellularity_index': np.zeros((X, Y, Z)),  # Inflammation
            'edema_index': np.zeros((X, Y, Z)),        # Edema
            'fiber_density': np.zeros((X, Y, Z)),      # Fiber density
            
            # Fitting quality
            'R2': np.zeros((X, Y, Z)),
            'S0': np.zeros((X, Y, Z))
        }
        
        # Count voxels to process
        n_voxels = np.sum(mask)
        
        # Fit each voxel with progress bar
        if show_progress:
            progress_bar = tqdm(total=n_voxels, desc="Fitting DBSI", 
                              position=0, leave=True, file=sys.stdout,
                              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        else:
            progress_bar = None
        
        voxel_count = 0
        for x in range(X):
            for y in range(Y):
                for z in range(Z):
                    if not bool(mask[x, y, z]):
                        continue
                    
                    signal = volume[x, y, z, :]
                    params, info = self.fit_voxel(signal, bvals, bvecs_norm, method)
                    
                    if info['success']:
                        # Base parameters
                        param_maps['f_restricted'][x, y, z] = params[0]
                        param_maps['D_restricted'][x, y, z] = params[1]
                        param_maps['f_hindered'][x, y, z] = params[2]
                        param_maps['D_hindered'][x, y, z] = params[3]
                        param_maps['f_water'][x, y, z] = params[4]
                        param_maps['f_fiber'][x, y, z] = params[5]
                        param_maps['D_axial'][x, y, z] = params[6]
                        param_maps['D_radial'][x, y, z] = params[7]
                        param_maps['theta'][x, y, z] = params[8]
                        param_maps['phi'][x, y, z] = params[9]
                        
                        # Create DBSIParams object to calculate derived metrics
                        dbsi_params = DBSIParams(
                            f_restricted=params[0], 
                            D_restricted=params[1], 
                            f_hindered=params[2], 
                            D_hindered=params[3],
                            f_water=params[4], 
                            D_water=3.0e-3,
                            f_fiber=params[5], 
                            D_axial=params[6], 
                            D_radial=params[7],
                            theta=params[8], 
                            phi=params[9]
                        )
                        
                        # Aggregated fractions
                        param_maps['f_iso'][x, y, z] = dbsi_params.f_iso_total
                        param_maps['f_aniso'][x, y, z] = dbsi_params.f_aniso_total
                        
                        # Anisotropy and diffusivity metrics
                        param_maps['fiber_FA'][x, y, z] = dbsi_params.fiber_FA
                        param_maps['fiber_MD'][x, y, z] = dbsi_params.fiber_MD
                        param_maps['overall_FA'][x, y, z] = dbsi_params.overall_FA
                        param_maps['overall_MD'][x, y, z] = dbsi_params.overall_MD
                        param_maps['FA'][x, y, z] = dbsi_params.overall_FA  # Alias
                        param_maps['MD'][x, y, z] = dbsi_params.overall_MD  # Alias
                        param_maps['AD'][x, y, z] = dbsi_params.axial_diffusivity
                        param_maps['RD'][x, y, z] = dbsi_params.radial_diffusivity
                        
                        # Pathological indices
                        param_maps['cellularity_index'][x, y, z] = dbsi_params.cellularity_index
                        param_maps['edema_index'][x, y, z] = dbsi_params.edema_index
                        param_maps['fiber_density'][x, y, z] = dbsi_params.fiber_density
                        
                        # Quality
                        param_maps['R2'][x, y, z] = info['r_squared']
                        param_maps['S0'][x, y, z] = info['S0']
                    
                    voxel_count += 1
                    if progress_bar is not None:
                        progress_bar.update(1)
        
        if progress_bar is not None:
            progress_bar.close()
        
        return param_maps