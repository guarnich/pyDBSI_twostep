# dbsi_toolbox/twostep.py

import numpy as np
from typing import Dict, Optional
from .base import BaseDBSI
from .linear import DBSI_Linear
from .nonlinear import DBSI_NonLinear
from .common import DBSIParams

class DBSI_TwoStep(BaseDBSI):
    """
    Orchestrator class that implements the DBSI Two-Step approach:
    1. Run Linear DBSI (NNLS) to find active fiber compartments and spectrum.
    2. Use Linear results as initial guess for Non-Linear DBSI (Least Squares).
    
    This provides the robustness of Basis Spectrum with the precision of Non-Linear fitting.
    """
    
    def __init__(self, 
                 # Linear Params
                 iso_diffusivity_range=(0.0, 3.0e-3),
                 n_iso_bases=20,
                 reg_lambda=0.01,
                 filter_threshold=0.01,
                 # Shared Params
                 axial_diff_basis=1.5e-3,
                 radial_diff_basis=0.3e-3):
        
        # Initialize the two internal solvers
        self.linear_model = DBSI_Linear(
            iso_diffusivity_range=iso_diffusivity_range,
            n_iso_bases=n_iso_bases,
            axial_diff_basis=axial_diff_basis,
            radial_diff_basis=radial_diff_basis,
            reg_lambda=reg_lambda,
            filter_threshold=filter_threshold
        )
        
        self.nonlinear_model = DBSI_NonLinear() # Doesn't need config params at init
        
    def fit_volume(self, volume, bvals, bvecs, **kwargs):
        """
        Overrides fit_volume to setup the Linear Design Matrix first.
        """
        # 1. Setup standard bvals/bvecs for parent class logic
        flat_bvals = np.array(bvals).flatten()
        N = len(flat_bvals)
        
        if bvecs.shape == (3, N):
            current_bvecs = bvecs.T
        else:
            current_bvecs = bvecs
            
        # 2. IMPORTANT: Initialize the Linear Model's Matrix
        print("Step 1/2: Pre-calculating Linear Design Matrix...", end="")
        self.linear_model._build_design_matrix(flat_bvals, current_bvecs)
        # Share the current bvecs with both models
        self.linear_model.current_bvecs = current_bvecs
        self.nonlinear_model.current_bvecs = current_bvecs
        print(" Done.")
        
        print("Step 2/2: Running Two-Step Fitting (Linear -> NonLinear)...")
        # 3. Run the standard loop defined in BaseDBSI
        return super().fit_volume(volume, bvals, bvecs, **kwargs)

    def fit_voxel(self, signal: np.ndarray, bvals: np.ndarray) -> DBSIParams:
        """
        The core Two-Step logic for a single voxel.
        """
        # --- STEP 1: Linear Fit (Basis Spectrum) ---
        # This is fast and robust against local minima
        linear_result = self.linear_model.fit_voxel(signal, bvals)
        
        # If Linear fit failed completely (e.g. bad data), skip NonLinear
        if linear_result.f_fiber == 0 and linear_result.f_iso_total == 0:
            return linear_result
            
        # --- STEP 2: Non-Linear Fit (Refinement) ---
        # Use linear result as initial guess (p0) to refine diffusivities and angles
        # This extracts specific Axial/Radial diffusivities instead of fixed basis values
        final_result = self.nonlinear_model.fit_voxel(signal, bvals, initial_guess=linear_result)
        
        return final_result