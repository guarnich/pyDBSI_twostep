# dbsi_toolbox/calibration.py

import numpy as np
from typing import List, Dict, Optional, Tuple
from .twostep import DBSI_TwoStep

def generate_synthetic_signal_rician(
    bvals: np.ndarray, 
    bvecs: np.ndarray, 
    f_fiber: float, 
    f_cell: float, 
    f_water: float, 
    snr: float,
    physio_params: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Generates a synthetic diffusion signal with Rician noise distribution.
    
    Args:
        bvals: Array of b-values
        bvecs: Array of gradient directions (N, 3)
        f_fiber: Fiber fraction (0-1)
        f_cell: Restricted fraction (0-1)
        f_water: Free water fraction (0-1)
        snr: Signal-to-noise ratio
        physio_params: Optional dict with physiological parameters.
                       If None, uses literature-based defaults with realistic variation.
    
    Returns:
        Synthetic noisy signal normalized to S0=1
        
    References:
        Wang, Y. et al. (2011). Quantification of increased cellularity during 
        inflammatory demyelination. Brain, 134(12), 3590-3601.
        
        Guglielmetti, C. et al. (2016). Diffusion kurtosis imaging probes 
        cortical alterations and white matter pathology following cuprizone 
        induced demyelination and spontaneous remyelination. NeuroImage, 125, 363-377.
    """
    # 1. Physiological Parameter Sampling
    if physio_params is None:
        # Literature-based ranges with modest variation to simulate biological heterogeneity
        # Values from Wang et al. 2011, Cross & Song 2017, Alexander et al. 2007
        
        # Fiber diffusivities (white matter)
        # Healthy WM: AD ~1.2-1.9 μm²/ms, RD ~0.2-0.5 μm²/ms
        D_fiber_ax = np.random.uniform(1.2e-3, 1.8e-3)
        D_fiber_rad = np.random.uniform(0.2e-3, 0.5e-3)
        
        # Restricted compartment (cells, inflammation, dense tissue)
        # Near-zero diffusion, but not exactly 0 for numerical stability
        D_cell = np.random.uniform(0.0, 0.0003)
        
        # Free water (CSF, edema)
        # Temperature-dependent: 2.5-3.5 μm²/ms at body temp
        D_water = np.random.uniform(2.5e-3, 3.5e-3)
    else:
        D_fiber_ax = physio_params.get('D_fiber_ax', 1.5e-3)
        D_fiber_rad = physio_params.get('D_fiber_rad', 0.3e-3)
        D_cell = physio_params.get('D_cell', 0.0)
        D_water = physio_params.get('D_water', 3.0e-3)
    
    # 2. Random Fiber Orientation
    # Generate isotropic-distributed fiber direction
    theta = np.arccos(2 * np.random.random() - 1)  # Polar angle
    phi = 2 * np.pi * np.random.random()           # Azimuthal angle
    
    fiber_dir = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    
    # 3. Forward Model: Multi-Compartment Signal
    n_meas = len(bvals)
    signal = np.zeros(n_meas)
    
    for i in range(n_meas):
        # Angle-dependent anisotropic diffusion
        cos_angle = np.dot(bvecs[i], fiber_dir)
        D_app = D_fiber_rad + (D_fiber_ax - D_fiber_rad) * (cos_angle**2)
        
        # Individual compartment signals
        s_fiber = np.exp(-bvals[i] * D_app)
        s_cell = np.exp(-bvals[i] * D_cell)
        s_water = np.exp(-bvals[i] * D_water)
        
        # Weighted sum (normalized fractions)
        f_total = f_fiber + f_cell + f_water
        signal[i] = (f_fiber * s_fiber + f_cell * s_cell + f_water * s_water) / f_total
    
    # 4. Rician Noise Addition
    # MRI magnitude images follow Rician distribution
    # Simulate as magnitude of complex Gaussian noise
    sigma = 1.0 / snr
    noise_real = np.random.normal(0, sigma, n_meas)
    noise_imag = np.random.normal(0, sigma, n_meas)
    
    signal_noisy = np.sqrt((signal + noise_real)**2 + noise_imag**2)
    
    return signal_noisy


def optimize_dbsi_params(
    real_bvals: np.ndarray,
    real_bvecs: np.ndarray,
    snr_estimate: float = 20.0,
    n_monte_carlo: int = 1000,
    bases_grid: List[int] = [25, 50, 75, 100],
    lambdas_grid: List[float] = [0.01, 0.1, 0.25, 0.5],
    ground_truth: Dict[str, float] = None,
    physio_ranges: Dict[str, Tuple[float, float]] = None,
    verbose: bool = True,
    seed: Optional[int] = 42
) -> Dict:
    """
    Monte Carlo calibration to optimize DBSI hyperparameters for a specific acquisition protocol.
    
    This function simulates realistic diffusion signals with known ground truth and noise,
    then evaluates how well different (n_bases, lambda) combinations recover the true parameters.
    
    Args:
        real_bvals: Array of b-values from the actual protocol
        real_bvecs: Array of b-vecs (N, 3) from the actual protocol
        snr_estimate: Estimated SNR of the real images
        n_monte_carlo: Number of noise realizations per configuration (higher = more robust)
        bases_grid: Candidate values for n_iso_bases to test
        lambdas_grid: Candidate values for reg_lambda to test
        ground_truth: Target fractions for phantom {'f_fiber', 'f_cell', 'f_water'}
                      If None, uses pathological scenario (30% inflammation)
        physio_ranges: Optional ranges for diffusivity parameters
                       Format: {'D_fiber_ax': (min, max), 'D_fiber_rad': (min, max), ...}
        verbose: Print detailed progress
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing optimal parameters and performance metrics
        
    Scientific Justification:
        Unlike DTI (which has closed-form solutions), DBSI's regularization and basis
        discretization introduce protocol-dependent bias-variance tradeoffs. This calibration
        ensures the model is tuned to YOUR specific b-values, SNR, and shell structure.
        
    Reference:
        Similar methodology used in:
        Novikov, D.S. et al. (2019). Quantifying brain microstructure with diffusion MRI: 
        Theory and parameter estimation. NMR in Biomedicine, 32(4), e3998.
    """
    
    # Set seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        if verbose:
            print(f"[Calibration] Random seed: {seed}")
    
    # Default ground truth: simulates a lesion with inflammation
    if ground_truth is None:
        ground_truth = {
            'f_fiber': 0.5,   # 50% intact fibers
            'f_cell': 0.3,    # 30% inflammation/cellularity (target metric)
            'f_water': 0.2    # 20% edema/free water
        }
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"[Calibration] Protocol-Specific Hyperparameter Optimization")
        print(f"{'='*80}")
        print(f"  Protocol: {len(real_bvals)} volumes")
        print(f"  Target SNR: {snr_estimate:.1f}")
        print(f"  Monte Carlo iterations: {n_monte_carlo}")
        print(f"  Ground Truth: Fiber={ground_truth['f_fiber']:.2f}, "
              f"Cell={ground_truth['f_cell']:.2f}, Water={ground_truth['f_water']:.2f}")
        print(f"{'-'*80}")
        print(f"{'Bases':<6} | {'Lambda':<8} | {'Mean Est':<10} | {'MAE':<10} | {'Std Dev':<10} | {'Bias':<10}")
        print(f"{'-'*80}")

    results = []

    # Standardize vectors
    flat_bvals = np.array(real_bvals).flatten()
    if real_bvecs.shape[0] == 3 and real_bvecs.shape[1] != 3:
        clean_bvecs = real_bvecs.T
    else:
        clean_bvecs = real_bvecs
    
    # Normalize gradient directions
    clean_bvecs = clean_bvecs / (np.linalg.norm(clean_bvecs, axis=1, keepdims=True) + 1e-10)

    # --- GRID SEARCH LOOP ---
    for n_bases in bases_grid:
        for reg in lambdas_grid:
            
            errors = []
            estimates = []
            
            # Initialize Model
            model = DBSI_TwoStep(
                n_iso_bases=n_bases,
                reg_lambda=reg,
                iso_diffusivity_range=(0.0, 3.0e-3)
            )
            
            # Pre-build design matrix (speeds up fitting)
            model.spectrum_model.design_matrix = model.spectrum_model._build_design_matrix(
                flat_bvals, clean_bvecs
            )
            model.spectrum_model.current_bvecs = clean_bvecs
            model.fitting_model.current_bvecs = clean_bvecs
            
            # --- MONTE CARLO SIMULATION ---
            for mc_iter in range(n_monte_carlo):
                # Generate synthetic signal with random physiological variation
                physio_params = None
                if physio_ranges is not None:
                    physio_params = {
                        k: np.random.uniform(v[0], v[1]) 
                        for k, v in physio_ranges.items()
                    }
                
                sig = generate_synthetic_signal_rician(
                    flat_bvals, clean_bvecs,
                    ground_truth['f_fiber'], 
                    ground_truth['f_cell'], 
                    ground_truth['f_water'],
                    snr=snr_estimate,
                    physio_params=physio_params
                )
                
                try:
                    res = model.fit_voxel(sig, flat_bvals)
                    
                    # Primary metric: restricted fraction (cellularity)
                    estimates.append(res.f_restricted)
                    errors.append(abs(res.f_restricted - ground_truth['f_cell']))
                    
                except Exception:
                    # Skip failed fits (rare, but can happen with extreme noise)
                    pass

            if not errors:
                continue  # No successful fits for this config

            # --- COMPUTE STATISTICS ---
            avg_error = np.mean(errors)          # Mean Absolute Error
            avg_estimate = np.mean(estimates)    # Mean estimate
            std_dev = np.std(errors)             # Standard deviation
            bias = avg_estimate - ground_truth['f_cell']  # Systematic bias
            
            if verbose:
                print(f"{n_bases:<6} | {reg:<8.3f} | {avg_estimate:.4f}     | "
                      f"{avg_error*100:6.2f}%    | {std_dev*100:6.2f}%    | {bias:+.4f}")
            
            results.append({
                'n_bases': n_bases,
                'reg_lambda': reg,
                'avg_error': avg_error,
                'std_dev': std_dev,
                'avg_estimate': avg_estimate,
                'bias': bias,
                'rmse': np.sqrt(np.mean(np.array(errors)**2))
            })

    # --- SELECT OPTIMAL CONFIGURATION ---
    # Criterion: Minimize MAE (primary) with penalty for high variance
    # Alternative criteria: minimize RMSE, minimize bias, etc.
    best_config = min(results, key=lambda x: x['avg_error'] + 0.5 * x['std_dev'])
    
    if verbose:
        print(f"{'-'*80}")
        print(f"[Calibration] OPTIMAL CONFIGURATION:")
        print(f"  Bases: {best_config['n_bases']}")
        print(f"  Lambda: {best_config['reg_lambda']}")
        print(f"  Mean Absolute Error: {best_config['avg_error']*100:.2f}%")
        print(f"  Bias: {best_config['bias']:+.4f}")
        print(f"  Standard Deviation: {best_config['std_dev']*100:.2f}%")
        print(f"  RMSE: {best_config['rmse']*100:.2f}%")
        print(f"{'='*80}\n")
        
    return best_config


def evaluate_calibration_stability(
    real_bvals: np.ndarray,
    real_bvecs: np.ndarray,
    optimal_params: Dict,
    snr_estimate: float,
    n_trials: int = 100,
    verbose: bool = True
) -> Dict:
    """
    Evaluates the stability of the calibrated parameters across different noise realizations.
    This provides confidence intervals for the optimal configuration.
    
    Args:
        real_bvals: Acquisition b-values
        real_bvecs: Acquisition gradient directions
        optimal_params: Dictionary from optimize_dbsi_params()
        snr_estimate: Estimated SNR
        n_trials: Number of test trials
        verbose: Print results
        
    Returns:
        Dictionary with stability metrics
    """
    if verbose:
        print("[Calibration] Evaluating stability of optimal parameters...")
    
    flat_bvals = np.array(real_bvals).flatten()
    clean_bvecs = real_bvecs.T if real_bvecs.shape[0] == 3 else real_bvecs
    
    model = DBSI_TwoStep(
        n_iso_bases=optimal_params['n_bases'],
        reg_lambda=optimal_params['reg_lambda'],
        iso_diffusivity_range=(0.0, 3.0e-3)
    )
    
    model.spectrum_model.design_matrix = model.spectrum_model._build_design_matrix(
        flat_bvals, clean_bvecs
    )
    model.spectrum_model.current_bvecs = clean_bvecs
    model.fitting_model.current_bvecs = clean_bvecs
    
    ground_truth = {'f_fiber': 0.5, 'f_cell': 0.3, 'f_water': 0.2}
    
    estimates = []
    for _ in range(n_trials):
        sig = generate_synthetic_signal_rician(
            flat_bvals, clean_bvecs,
            ground_truth['f_fiber'], ground_truth['f_cell'], ground_truth['f_water'],
            snr=snr_estimate
        )
        
        try:
            res = model.fit_voxel(sig, flat_bvals)
            estimates.append(res.f_restricted)
        except:
            pass
    
    estimates = np.array(estimates)
    
    stability = {
        'mean': np.mean(estimates),
        'std': np.std(estimates),
        'cv': np.std(estimates) / (np.mean(estimates) + 1e-10),  # Coefficient of variation
        'ci_95': (np.percentile(estimates, 2.5), np.percentile(estimates, 97.5)),
        'bias': np.mean(estimates) - ground_truth['f_cell']
    }
    
    if verbose:
        print(f"  Mean estimate: {stability['mean']:.4f}")
        print(f"  Standard deviation: {stability['std']:.4f}")
        print(f"  95% CI: [{stability['ci_95'][0]:.4f}, {stability['ci_95'][1]:.4f}]")
        print(f"  Coefficient of Variation: {stability['cv']*100:.2f}%")
        print(f"  Bias: {stability['bias']:+.4f}")
    
    return stability