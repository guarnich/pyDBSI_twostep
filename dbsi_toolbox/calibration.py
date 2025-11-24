# dbsi_toolbox/calibration.py

import numpy as np
from typing import List, Dict, Optional
from .twostep import DBSI_TwoStep

def generate_synthetic_signal_rician(
    bvals: np.ndarray, 
    bvecs: np.ndarray, 
    f_fiber: float, 
    f_cell: float, 
    f_water: float, 
    snr: float
) -> np.ndarray:
    """
    Generates a synthetic diffusion signal with Rician noise distribution.
    Physiological parameters are based on Wang et al., 2011. 
    """
    # 1. Standard physiological parameters
    D_fiber_ax = 1.5e-3
    D_fiber_rad = 0.3e-3
    D_cell = 0.0e-3      # Restricted diffusion (Cells/Inflammation)
    D_water = 3.0e-3     # Free diffusion (CSF/Edema)
    
    # Arbitrary fiber direction along X-axis
    fiber_dir = np.array([1.0, 0.0, 0.0])
    
    # 2. Forward Model Calculation
    n_meas = len(bvals)
    signal = np.zeros(n_meas)
    
    for i in range(n_meas):
        cos_angle = np.dot(bvecs[i], fiber_dir)
        D_app = D_fiber_rad + (D_fiber_ax - D_fiber_rad) * (cos_angle**2)
        
        s_fiber = np.exp(-bvals[i] * D_app)
        s_cell = np.exp(-bvals[i] * D_cell)
        s_water = np.exp(-bvals[i] * D_water)
        
        signal[i] = (f_fiber * s_fiber) + (f_cell * s_cell) + (f_water * s_water)
    
    # 3. Add Rician Noise
    # Rician noise is simulated as the magnitude of a complex signal 
    # with Gaussian noise added to both real and imaginary parts.
    sigma = 1.0 / snr
    noise_real = np.random.normal(0, sigma, n_meas)
    noise_imag = np.random.normal(0, sigma, n_meas)
    
    signal_noisy = np.sqrt((signal + noise_real)**2 + noise_imag**2)
    
    return signal_noisy


def optimize_dbsi_params(
    real_bvals: np.ndarray,
    real_bvecs: np.ndarray,
    snr_estimate: float = 30.0,
    n_monte_carlo: int = 500,
    bases_grid: List[int] = [25, 50, 75],
    lambdas_grid: List[float] = [0.01, 0.1, 0.5],
    ground_truth: Dict[str, float] = {'f_fiber': 0.5, 'f_cell': 0.3, 'f_water': 0.2},
    verbose: bool = True,
    seed: Optional[int] = 42
) -> Dict:
    """
    Performs a Monte Carlo calibration to identify the optimal DBSI hyperparameters
    (n_iso_bases, reg_lambda) specific to the provided acquisition protocol.
    
    Args:
        real_bvals: Array of b-values from the real protocol.
        real_bvecs: Array of b-vecs from the real protocol (N, 3).
        snr_estimate: Estimated SNR of the real images (default 30 for 3T).
        n_monte_carlo: Number of noise iterations per configuration (recommended >100).
        bases_grid: List of isotropic basis counts to test.
        lambdas_grid: List of regularization weights to test.
        ground_truth: Dictionary containing "true" fractions for the phantom.
        verbose: If True, prints progress to stdout.
        seed: Integer to seed the random number generator for reproducibility (default: 42).
        
    Returns:
        A dictionary containing the optimal parameters ('n_bases', 'reg_lambda') 
        and error statistics.
    """
    
    # Set seed for reproducibility
    if seed is not None:
        if verbose:
            print(f"[Calibration] Setting random seed to {seed} for reproducibility.")
        np.random.seed(seed)
    
    if verbose:
        print(f"\n[Calibration] Starting protocol optimization ({len(real_bvals)} volumes).")
        print(f"[Calibration] Monte Carlo ({n_monte_carlo} iter). Target Cell Fraction: {ground_truth['f_cell']}")
        print("-" * 80)
        print(f"{'Bases':<6} | {'Lambda':<8} | {'Avg Cell':<10} | {'Avg Error':<10} | {'Std Dev':<10}")
        print("-" * 80)

    results = []

    # Standardize vectors
    flat_bvals = np.array(real_bvals).flatten()
    # Ensure shape (N, 3) for signal generation
    if real_bvecs.shape[0] == 3 and real_bvecs.shape[1] != 3:
        clean_bvecs = real_bvecs.T
    else:
        clean_bvecs = real_bvecs

    # Grid Search Loop
    for n_bases in bases_grid:
        for reg in lambdas_grid:
            
            errors = []
            estimates = []
            
            # Initialize Model (Bypassing fit_volume for speed)
            # We focus on the Linear Spectrum step where lambda/bases are critical.
            model = DBSI_TwoStep(
                n_iso_bases=n_bases,
                reg_lambda=reg,
                iso_diffusivity_range=(0.0, 3.0e-3)
            )
            
            # Pre-calculate Design Matrix
            model.spectrum_model.design_matrix = model.spectrum_model._build_design_matrix(flat_bvals, clean_bvecs)
            model.spectrum_model.current_bvecs = clean_bvecs
            model.fitting_model.current_bvecs = clean_bvecs
            
            # Monte Carlo Simulation Loop
            for _ in range(n_monte_carlo):
                # Generate fresh signal with new noise instance
                # (Since seed is set, this sequence will be identical across runs)
                sig = generate_synthetic_signal_rician(
                    flat_bvals, clean_bvecs,
                    ground_truth['f_fiber'], ground_truth['f_cell'], ground_truth['f_water'],
                    snr=snr_estimate
                )
                
                try:
                    res = model.fit_voxel(sig, flat_bvals)
                    estimates.append(res.f_restricted)
                    errors.append(abs(res.f_restricted - ground_truth['f_cell']))
                except Exception:
                    pass # Skip failed fits (rare)

            if not errors: continue

            avg_error = np.mean(errors)
            avg_estimate = np.mean(estimates)
            std_dev = np.std(errors)
            
            if verbose:
                print(f"{n_bases:<6} | {reg:<8} | {avg_estimate:.4f}     | {avg_error*100:.2f}%      | {std_dev*100:.2f}%")
            
            results.append({
                'n_bases': n_bases,
                'reg_lambda': reg,
                'avg_error': avg_error,
                'std_dev': std_dev,
                'avg_estimate': avg_estimate
            })

    # Selection Criteria: Minimize Mean Absolute Error
    best_config = min(results, key=lambda x: x['avg_error'])
    
    if verbose:
        print("-" * 80)
        print(f"[Calibration] WINNER: {best_config['n_bases']} Bases, Lambda {best_config['reg_lambda']}")
        print(f"              Mean Error: {best_config['avg_error']*100:.2f}%")
        print("-" * 80)
        
    return best_config