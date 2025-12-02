#!/usr/bin/env python3
"""
examples/run_dbsi.py

Complete DBSI Two-Step Processing Pipeline with Enhanced Validation:
1. Data Loading & Protocol Validation
2. SNR Estimation (Rician-corrected)
3. Monte Carlo Hyperparameter Calibration
4. Volumetric Fitting with Progress Tracking
5. Quality Assessment & Results Export

Version: 0.3.0 (Scientific Grade)
"""

import argparse
import os
import time
import json
import numpy as np
import sys
import warnings

# Suppress non-critical warnings during processing
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Add parent directory to path if dbsi_toolbox is not installed globally
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from dbsi_toolbox.utils import (
        load_dwi_data_dipy, 
        save_parameter_maps, 
        estimate_snr,
        validate_acquisition_protocol
    )
    from dbsi_toolbox.calibration import (
        optimize_dbsi_params,
        evaluate_calibration_stability
    )
    from dbsi_toolbox.twostep import DBSI_TwoStep
except ImportError as e:
    print(f"[ERROR] Cannot import toolbox: {e}")
    print("Make sure to install the package: pip install -e .")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(
        description="DBSI Two-Step Pipeline with Advanced Calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (automatic calibration):
  python run_dbsi.py -i data.nii.gz -b data.bval -v data.bvec -m mask.nii.gz -o results/
  
  # Manual SNR specification:
  python run_dbsi.py -i data.nii.gz -b data.bval -v data.bvec -m mask.nii.gz -o results/ --snr 35
  
  # Extended calibration for high-quality data:
  python run_dbsi.py -i data.nii.gz -b data.bval -v data.bvec -m mask.nii.gz -o results/ --mc_iter 2000
  
  # Enable robust multi-start optimization:
  python run_dbsi.py -i data.nii.gz -b data.bval -v data.bvec -m mask.nii.gz -o results/ --multistart
        """
    )
    
    # === MANDATORY INPUTS ===
    required = parser.add_argument_group('Required Arguments')
    required.add_argument('--input', '-i', required=True, 
                         help='Path to 4D DWI NIfTI file (.nii or .nii.gz)')
    required.add_argument('--bval', '-b', required=True, 
                         help='Path to b-values file (.bval)')
    required.add_argument('--bvec', '-v', required=True, 
                         help='Path to b-vectors file (.bvec)')
    required.add_argument('--mask', '-m', required=True, 
                         help='Path to brain mask NIfTI file (MANDATORY)')
    required.add_argument('--out', '-o', required=True, 
                         help='Output directory for parameter maps and metadata')
    
    # === SNR OPTIONS ===
    snr_group = parser.add_argument_group('SNR Estimation')
    snr_group.add_argument('--snr', type=float, default=None, 
                          help='Manual SNR override. If not specified, estimated from data')
    snr_group.add_argument('--snr_method', choices=['auto', 'temporal', 'spatial'], 
                          default='auto',
                          help='SNR estimation method (default: auto-select based on data)')
    
    # === CALIBRATION OPTIONS ===
    calib_group = parser.add_argument_group('Calibration Parameters')
    calib_group.add_argument('--mc_iter', type=int, default=1000, 
                            help='Monte Carlo iterations for calibration (default: 1000, recommended: 500-2000)')
    calib_group.add_argument('--skip_calibration', action='store_true',
                            help='Skip calibration and use default parameters (faster but less accurate)')
    calib_group.add_argument('--manual_bases', type=int, default=50,
                            help='Manual n_iso_bases if skipping calibration (default: 50)')
    calib_group.add_argument('--manual_lambda', type=float, default=0.1,
                            help='Manual reg_lambda if skipping calibration (default: 0.1)')
    calib_group.add_argument('--stability_check', action='store_true',
                            help='Run stability analysis after calibration (adds ~30sec)')
    
    # === FITTING OPTIONS ===
    fit_group = parser.add_argument_group('Fitting Parameters')
    fit_group.add_argument('--multistart', action='store_true',
                          help='Enable multi-start optimization for robust NLLS fitting (slower but more robust)')
    fit_group.add_argument('--multistart_threshold', type=float, default=0.1,
                          help='Fiber fraction threshold for triggering multi-start (default: 0.1)')
    
    # === ADVANCED OPTIONS ===
    advanced = parser.add_argument_group('Advanced Options')
    advanced.add_argument('--seed', type=int, default=42,
                         help='Random seed for reproducibility (default: 42)')
    advanced.add_argument('--verbose', action='store_true',
                         help='Print detailed progress information')
    
    return parser.parse_args()

def print_header():
    """Prints a nice header for the pipeline."""
    print("\n" + "="*80)
    print(" " * 20 + "DBSI TWO-STEP PROCESSING PIPELINE")
    print(" " * 25 + "Version 0.3.0 (Scientific)")
    print("="*80 + "\n")

def main():
    args = parse_args()
    start_time = time.time()
    
    print_header()

    # =======================================================================
    # STEP 1: DATA LOADING & VALIDATION
    # =======================================================================
    print(f"[Step 1/5] Loading and Validating Data")
    print("-" * 80)
    
    if not os.path.exists(args.out):
        os.makedirs(args.out)
        print(f"✓ Created output directory: {args.out}")

    try:
        dwi_data, affine, gtab, mask = load_dwi_data_dipy(
            args.input, args.bval, args.bvec, args.mask
        )
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Data loading failed: {e}")
        sys.exit(1)
    
    # Protocol validation
    protocol_info = validate_acquisition_protocol(gtab, verbose=True)
    
    print(f"\n✓ Data loaded successfully")
    print(f"  Volume shape: {dwi_data.shape}")
    print(f"  Brain voxels: {np.sum(mask):,}")
    print(f"  Estimated processing time: ~{np.sum(mask) * 0.01 / 60:.1f} minutes")

    # =======================================================================
    # STEP 2: SNR ESTIMATION
    # =======================================================================
    print(f"\n[Step 2/5] Signal-to-Noise Ratio Estimation")
    print("-" * 80)
    
    if args.snr is not None:
        estimated_snr = args.snr
        snr_source = "user_manual"
        print(f"✓ Using user-specified SNR: {estimated_snr:.2f}")
    else:
        estimated_snr = estimate_snr(
            dwi_data, gtab, affine, mask, 
            method=args.snr_method
        )
        snr_source = f"auto_{args.snr_method}"
        print(f"✓ Estimated SNR: {estimated_snr:.2f}")
    
    # SNR quality warning
    if estimated_snr < 15:
        print(f"\n⚠ WARNING: Low SNR detected ({estimated_snr:.1f})")
        print(f"  This may result in noisy parameter estimates")
        print(f"  Consider using higher Monte Carlo iterations (--mc_iter 2000)")

    # =======================================================================
    # STEP 3: HYPERPARAMETER CALIBRATION
    # =======================================================================
    print(f"\n[Step 3/5] Hyperparameter Calibration")
    print("-" * 80)
    
    if args.skip_calibration:
        print("⚠ Skipping calibration (using manual parameters)")
        optimal_bases = args.manual_bases
        optimal_lambda = args.manual_lambda
        calibration_results = {
            'method': 'manual',
            'n_bases': optimal_bases,
            'reg_lambda': optimal_lambda
        }
    else:
        print(f"Running Monte Carlo optimization...")
        print(f"  Iterations: {args.mc_iter}")
        print(f"  Random seed: {args.seed}")
        
        # Search grids (can be customized based on protocol)
        if protocol_info['is_multishell']:
            bases_grid = [30, 50, 75, 100]
            print(f"  Multi-shell detected: testing finer basis resolution")
        else:
            bases_grid = [20, 40, 60, 80]
        
        lambdas_grid = [0.01, 0.05, 0.1, 0.25, 0.5]
        
        best_params = optimize_dbsi_params(
            gtab.bvals, 
            gtab.bvecs,
            snr_estimate=estimated_snr,
            n_monte_carlo=args.mc_iter,
            bases_grid=bases_grid,
            lambdas_grid=lambdas_grid,
            verbose=args.verbose,
            seed=args.seed
        )
        
        optimal_bases = best_params['n_bases']
        optimal_lambda = best_params['reg_lambda']
        calibration_results = best_params
        
        print(f"\n✓ Optimal configuration identified:")
        print(f"  Isotropic bases: {optimal_bases}")
        print(f"  Regularization λ: {optimal_lambda}")
        print(f"  Expected MAE: {best_params['avg_error']*100:.2f}%")
        
        # Optional stability check
        if args.stability_check:
            print(f"\nRunning stability analysis...")
            stability = evaluate_calibration_stability(
                gtab.bvals, gtab.bvecs, best_params, estimated_snr,
                n_trials=100, verbose=True
            )
            calibration_results['stability'] = stability

    # =======================================================================
    # STEP 4: VOLUMETRIC FITTING
    # =======================================================================
    print(f"\n[Step 4/5] DBSI Two-Step Fitting")
    print("-" * 80)
    
    model = DBSI_TwoStep(
        n_iso_bases=optimal_bases,
        reg_lambda=optimal_lambda,
        iso_diffusivity_range=(0.0, 3.0e-3),
        filter_threshold=0.01
    )
    
    # Enable multi-start if requested
    if args.multistart:
        print(f"✓ Multi-start optimization enabled")
        print(f"  Trigger threshold: fiber fraction < {args.multistart_threshold}")
        model.fitting_model.use_multistart = True
        model.fitting_model.multistart_threshold = args.multistart_threshold
    
    print(f"\nStarting voxel-wise fitting...")
    print(f"(This will take approximately {np.sum(mask) * 0.01 / 60:.1f} minutes)\n")
    
    fit_start = time.time()
    maps = model.fit_volume(
        dwi_data, 
        gtab.bvals, 
        gtab.bvecs, 
        mask=mask, 
        show_progress=True
    )
    fit_time = time.time() - fit_start
    
    print(f"\n✓ Fitting completed in {fit_time/60:.2f} minutes")
    print(f"  Average time per voxel: {fit_time/np.sum(mask)*1000:.2f} ms")

    # =======================================================================
    # STEP 5: QUALITY ASSESSMENT & SAVING
    # =======================================================================
    print(f"\n[Step 5/5] Quality Assessment and Export")
    print("-" * 80)
    
    # Basic quality metrics
    r2_mean = np.mean(maps['r_squared'][mask])
    r2_median = np.median(maps['r_squared'][mask])
    
    fiber_mean = np.mean(maps['fiber_fraction'][mask])
    restricted_mean = np.mean(maps['restricted_fraction'][mask])
    
    print(f"Quality Metrics:")
    print(f"  Mean R²: {r2_mean:.4f}")
    print(f"  Median R²: {r2_median:.4f}")
    print(f"  Mean Fiber Fraction: {fiber_mean:.4f}")
    print(f"  Mean Restricted Fraction: {restricted_mean:.4f}")
    
    if r2_median < 0.7:
        print(f"\n⚠ WARNING: Low median R² ({r2_median:.3f})")
        print(f"  This may indicate poor model fit or data quality issues")
    
    # Save parameter maps
    print(f"\nSaving results to: {args.out}")
    save_parameter_maps(maps, affine, output_dir=args.out, prefix='dbsi')
    
    # Save comprehensive metadata
    metadata = {
        "pipeline_version": "0.3.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": os.path.abspath(args.input),
        "processing_time_seconds": time.time() - start_time,
        "protocol": {
            "n_volumes": protocol_info['n_volumes'],
            "n_b0": protocol_info['n_b0'],
            "n_dwi": protocol_info['n_dwi'],
            "b_max": protocol_info['b_max'],
            "shells": protocol_info['shells'],
            "is_multishell": protocol_info['is_multishell'],
            "warnings": protocol_info['warnings']
        },
        "snr": {
            "value": float(estimated_snr),
            "source": snr_source,
            "method": args.snr_method
        },
        "calibration": calibration_results,
        "model_config": {
            "n_iso_bases": optimal_bases,
            "reg_lambda": optimal_lambda,
            "multistart_enabled": args.multistart,
            "iso_diffusivity_range": [0.0, 3.0e-3]
        },
        "quality_metrics": {
            "mean_r_squared": float(r2_mean),
            "median_r_squared": float(r2_median),
            "mean_fiber_fraction": float(fiber_mean),
            "mean_restricted_fraction": float(restricted_mean),
            "n_brain_voxels": int(np.sum(mask)),
            "fit_time_per_voxel_ms": float(fit_time / np.sum(mask) * 1000)
        },
        "reproducibility": {
            "random_seed": args.seed,
            "command_line": " ".join(sys.argv)
        }
    }
    
    meta_path = os.path.join(args.out, "dbsi_pipeline_info.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved to: {meta_path}")
    
    # Final summary
    print("\n" + "="*80)
    print(" " * 25 + "PROCESSING COMPLETED")
    print("="*80)
    print(f"Total time: {(time.time() - start_time)/60:.2f} minutes")
    print(f"Output directory: {args.out}")
    print(f"\nGenerated files:")
    print(f"  - 10 parameter maps (dbsi_*.nii.gz)")
    print(f"  - Pipeline metadata (dbsi_pipeline_info.json)")
    print("\n✓ Ready for analysis!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()