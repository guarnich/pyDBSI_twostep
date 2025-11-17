# dbsi_toolbox/model.py

import numpy as np
from scipy.optimize import nnls
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
from tqdm import tqdm
import sys
import warnings

@dataclass
class DBSIParams:
    """
    Dataclass for holding DBSI results for a single voxel.
    Aggregates weights from the spectral solution into clinical metrics.
    """
    # Isotropic Fractions (derived from the spectrum)
    f_restricted: float      # Restricted fraction (cells/inflammation, D <= 0.3)
    f_hindered: float        # Hindered fraction (extracellular space/edema, 0.3 < D <= 2.0)
    f_water: float           # Free water fraction (CSF, D > 2.0)
    
    # Anisotropic Fractions
    f_fiber: float           # Total anisotropic fraction (sum of all fiber weights)
    
    # Fiber Properties
    fiber_dir: np.ndarray    # Dominant fiber direction vector (x, y, z)
    axial_diffusivity: float # Axial diffusivity of the basis tensor
    radial_diffusivity: float # Radial diffusivity of the basis tensor
    
    # Fitting Quality
    r_squared: float         # Coefficient of determination (R^2)

    @property
    def f_iso_total(self) -> float:
        """Total isotropic fraction."""
        return self.f_restricted + self.f_hindered + self.f_water

    @property
    def fiber_density(self) -> float:
        """Normalized fiber density (Fiber Fraction)."""
        total = self.f_fiber + self.f_iso_total
        return self.f_fiber / total if total > 0 else 0.0

class DBSIModel:
    """
    Implementation of Diffusion Basis Spectrum Imaging (DBSI) using 
    Non-Negative Least Squares (NNLS).
    """
    
    def __init__(self, 
                 iso_diffusivity_range: Tuple[float, float] = (0.0, 3.0e-3),
                 n_iso_bases: int = 20,
                 axial_diff_basis: float = 1.5e-3,
                 radial_diff_basis: float = 0.3e-3):
        """
        Initialize the DBSI Model configuration.
        Note: Data (bvals/bvecs) are passed during fit_volume.
        """
        # Configuration parameters
        self.iso_range = iso_diffusivity_range
        self.n_iso_bases = n_iso_bases
        self.axial_diff_basis = axial_diff_basis
        self.radial_diff_basis = radial_diff_basis
        
        # Placeholders for data-dependent structures
        self.design_matrix = None
        self.iso_diffusivities = None
        self.current_bvecs = None
        
    def _build_design_matrix(self, bvals: np.ndarray, bvecs: np.ndarray) -> np.ndarray:
        """
        Constructs the dictionary matrix A based on input gradients.
        """
        n_meas = len(bvals)
        n_aniso = len(bvecs) 
        
        # Generate Isotropic Spectrum
        self.iso_diffusivities = np.linspace(
            self.iso_range[0], 
            self.iso_range[1], 
            self.n_iso_bases
        )
        n_iso = len(self.iso_diffusivities)
        
        # Initialize Matrix A
        A = np.zeros((n_meas, n_aniso + n_iso))
        
        # --- Part 1: Anisotropic Basis Functions (Fibers) ---
        for j in range(n_aniso):
            fiber_dir = bvecs[j]
            # Normalize fiber direction
            norm = np.linalg.norm(fiber_dir)
            if norm > 0:
                fiber_dir = fiber_dir / norm
                
            # Dot product of all gradient directions with the current fiber basis
            cos_angles = np.dot(bvecs, fiber_dir)
            
            # D_app = D_rad + (D_ax - D_rad) * (g . fiber_dir)^2
            D_app = self.radial_diff_basis + (self.axial_diff_basis - self.radial_diff_basis) * (cos_angles**2)
            
            A[:, j] = np.exp(-bvals * D_app)
            
        # --- Part 2: Isotropic Basis Functions (Spectrum) ---
        for i, D_iso in enumerate(self.iso_diffusivities):
            A[:, n_aniso + i] = np.exp(-bvals * D_iso)
            
        return A

    def fit_voxel(self, signal: np.ndarray, bvals: np.ndarray) -> DBSIParams:
        """
        Fits a single voxel using NNLS with the pre-calculated matrix.
        """
        if self.design_matrix is None:
            raise RuntimeError("Design matrix not built. Call fit_volume or internal build first.")

        # Signal Normalization (S / S0)
        if np.any(bvals < 50):
            S0 = np.mean(signal[bvals < 50])
        else:
            S0 = signal[0]
            
        if S0 <= 1e-6 or np.any(np.isnan(signal)):
            return self._get_empty_params()
            
        y = signal / S0
        
        # --- SOLVER: NNLS ---
        weights, _ = nnls(self.design_matrix, y)
        
        # --- Extract Metrics ---
        n_aniso = len(self.current_bvecs)
        
        # 1. Anisotropic Weights
        aniso_weights = weights[:n_aniso]
        f_fiber = np.sum(aniso_weights)
        
        # Dominant fiber direction
        if f_fiber > 0:
            dom_idx = np.argmax(aniso_weights)
            main_dir = self.current_bvecs[dom_idx]
        else:
            main_dir = np.array([0.0, 0.0, 0.0])
        
        # 2. Isotropic Weights
        iso_weights = weights[n_aniso:]
        
        # Aggregate spectrum compartments (Thresholds in mm^2/s)
        # Restricted (Cells): D <= 0.3e-3
        mask_res = self.iso_diffusivities <= 0.3e-3
        f_restricted = np.sum(iso_weights[mask_res])
        
        # Hindered (Tissue): 0.3e-3 < D <= 2.0e-3
        mask_hin = (self.iso_diffusivities > 0.3e-3) & (self.iso_diffusivities <= 2.0e-3)
        f_hindered = np.sum(iso_weights[mask_hin])
        
        # Free Water (CSF): D > 2.0e-3
        mask_wat = self.iso_diffusivities > 2.0e-3
        f_water = np.sum(iso_weights[mask_wat])
        
        # R-Squared
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

    def fit_volume(self, 
                   volume: np.ndarray, 
                   bvals: np.ndarray, 
                   bvecs: np.ndarray, 
                   mask: Optional[np.ndarray] = None,
                   method: str = 'least_squares', # Kept for API compatibility
                   show_progress: bool = True) -> Dict[str, np.ndarray]:
        """
        Fits the DBSI model to a 4D volume.
        Builds the design matrix once based on input bvals/bvecs, then iterates.
        """
        # Input validation and preparation
        if volume.ndim != 4:
            raise ValueError(f"Volume must be 4D, got {volume.ndim}")
        
        X, Y, Z, N = volume.shape
        
        # Handle bvals/bvecs shapes
        bvals = np.array(bvals).flatten()
        if len(bvals) != N:
             raise ValueError(f"bvals length ({len(bvals)}) mismatch volume depth ({N})")
             
        if bvecs.shape == (3, N):
            bvecs = bvecs.T
        elif bvecs.shape != (N, 3):
             raise ValueError(f"bvecs shape {bvecs.shape} mismatch. Expected ({N}, 3) or (3, {N})")
        
        # Store current vectors for direction recovery
        self.current_bvecs = bvecs
        
        # --- BUILD DESIGN MATRIX ONCE ---
        print("Pre-calculating DBSI Design Matrix...", end="")
        self.design_matrix = self._build_design_matrix(bvals, bvecs)
        print(" Done.")
        
        # Mask handling
        if mask is None:
            # Simple threshold mask if none provided
            b0_mean = np.mean(volume[..., bvals < 50], axis=-1)
            mask = b0_mean > np.percentile(b0_mean, 10)
            
        # Initialize output maps
        param_maps = {
            'fiber_fraction': np.zeros((X, Y, Z)),
            'restricted_fraction': np.zeros((X, Y, Z)),
            'hindered_fraction': np.zeros((X, Y, Z)),
            'water_fraction': np.zeros((X, Y, Z)),
            'fiber_dir_x': np.zeros((X, Y, Z)),
            'fiber_dir_y': np.zeros((X, Y, Z)),
            'fiber_dir_z': np.zeros((X, Y, Z)),
            'r_squared': np.zeros((X, Y, Z)),
        }
        
        n_voxels = np.sum(mask)
        
        # Setup progress bar
        pbar = tqdm(total=n_voxels, desc="Fitting DBSI", unit="vox", disable=not show_progress)
        
        # Voxel-wise fitting
        for x in range(X):
            for y in range(Y):
                for z in range(Z):
                    if not mask[x, y, z]:
                        continue
                    
                    signal = volume[x, y, z, :]
                    
                    # Fit
                    params = self.fit_voxel(signal, bvals)
                    
                    # Store results
                    param_maps['fiber_fraction'][x, y, z] = params.fiber_density
                    param_maps['restricted_fraction'][x, y, z] = params.f_restricted
                    param_maps['hindered_fraction'][x, y, z] = params.f_hindered
                    param_maps['water_fraction'][x, y, z] = params.f_water
                    param_maps['fiber_dir_x'][x, y, z] = params.fiber_dir[0]
                    param_maps['fiber_dir_y'][x, y, z] = params.fiber_dir[1]
                    param_maps['fiber_dir_z'][x, y, z] = params.fiber_dir[2]
                    param_maps['r_squared'][x, y, z] = params.r_squared
                    
                    pbar.update(1)
        
        pbar.close()
        return param_maps

    def _get_empty_params(self) -> DBSIParams:
        return DBSIParams(
            f_restricted=0.0, f_hindered=0.0, f_water=0.0, f_fiber=0.0,
            fiber_dir=np.zeros(3), axial_diffusivity=0.0, radial_diffusivity=0.0,
            r_squared=0.0
        )