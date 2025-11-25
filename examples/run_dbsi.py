#!/usr/bin/env python3
"""
examples/run_dbsi.py

This script demonstrates the complete DBSI Two-Step processing pipeline:
1. Data Loading & Preprocessing
2. Automatic SNR Estimation (with intelligent fallback logic)
3. Protocol-specific Hyperparameter Optimization (Monte Carlo Calibration)
4. Model Initialization and Volumetric Fitting
5. Saving Results and Metadata
"""

import argparse
import os
import time
import json
import numpy as np
import sys

# Add parent directory to path if dbsi_toolbox is not installed globally
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from dbsi_toolbox.utils import load_dwi_data_dipy, save_parameter_maps, estimate_snr
    from dbsi_toolbox.calibration import optimize_dbsi_params
    from dbsi_toolbox.twostep import DBSI_TwoStep
except ImportError as e:
    print(f"Error importing toolbox: {e}")
    print("Make sure to run the script from the project root or install the package.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Complete DBSI Pipeline (Auto-Calibration + Two-Step Fit)")
    
    # Mandatory Inputs
    parser.add_argument('--input', '-i', required=True, help='Path to 4D DWI NIfTI file')
    parser.add_argument('--bval', '-b', required=True, help='Path to .bval file')
    parser.add_argument('--bvec', '-v', required=True, help='Path to .bvec file')
    parser.add_argument('--mask', '-m', required=True, help='Path to Brain Mask NIfTI file')
    parser.add_argument('--out', '-o', required=True, help='Output directory')
    
    # Advanced Options
    parser.add_argument('--snr', type=float, default=None, 
                        help='Manual SNR override (if not specified, it is estimated from data)')
    parser.add_argument('--mc_iter', type=int, default=1000, 
                        help='Monte Carlo iterations for calibration (default: 1000)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    start_time = time.time()
    
    print("\n" + "="*50)
    print("   DBSI PROCESSING PIPELINE v0.2.0")
    print("="*50 + "\n")

    # -----------------------------------------------------------
    # 1. DATA LOADING
    # -----------------------------------------------------------
    print(f"[1/5] Loading Data...")
    
    if not os.path.exists(args.out):
        os.makedirs(args.out)
        print(f"      Created output directory: {args.out}")

    try:
        dwi_data, affine, gtab, mask = load_dwi_data_dipy(
            args.input, args.bval, args.bvec, args.mask
        )
    except Exception as e:
        print(f"[CRITICAL ERROR] Could not load data: {e}")
        sys.exit(1)
    
    real_bvals = gtab.bvals
    real_bvecs = gtab.bvecs
    print(f"      Volume Shape: {dwi_data.shape}")
    print(f"      Diffusion Directions: {len(real_bvals)}")

    # -----------------------------------------------------------
    # 2. SNR ESTIMATION
    # -----------------------------------------------------------
    print(f"\n[2/5] Estimating Signal-to-Noise Ratio (SNR)...")
    if args.snr is not None:
        estimated_snr = args.snr
        snr_source = "user_manual"
        print(f"      Using user-specified SNR: {estimated_snr}")
    else:
        # Use the intelligent function from utils.py
        estimated_snr = estimate_snr(dwi_data, gtab, affine, mask)
        snr_source = "auto_estimated"
        print(f"      Estimated/Used SNR: {estimated_snr:.2f}")

    # -----------------------------------------------------------
    # 3. CALIBRATION (Hyperparameter Optimization)
    # -----------------------------------------------------------
    print(f"\n[3/5] Calibrating Model Parameters...")
    print(f"      Running Monte Carlo optimization ({args.mc_iter} iterations)...")
    
    # Search Grid
    grid_bases = [20, 50, 75, 100]
    grid_lambdas = [0.01, 0.1, 0.25, 0.5]
    
    best_params = optimize_dbsi_params(
        real_bvals, 
        real_bvecs,
        snr_estimate=estimated_snr,
        n_monte_carlo=args.mc_iter,
        bases_grid=grid_bases,
        lambdas_grid=grid_lambdas,
        verbose=True
    )
    
    optimal_bases = best_params['n_bases']
    optimal_lambda = best_params['reg_lambda']
    
    print(f"      -> Optimal Configuration: Bases={optimal_bases}, Lambda={optimal_lambda}")

    # -----------------------------------------------------------
    # 4. FITTING (DBSI Two-Step)
    # -----------------------------------------------------------
    print(f"\n[4/5] Running DBSI Fit (Two-Step)...")
    print(f"      Initializing model with calibrated parameters.")
    
    model = DBSI_TwoStep(
        n_iso_bases=optimal_bases,
        reg_lambda=optimal_lambda,
        iso_diffusivity_range=(0.0, 3.0e-3),
        filter_threshold=0.01
    )
    
    # Fit the whole volume (with progress bar)
    maps = model.fit_volume(dwi_data, real_bvals, real_bvecs, mask=mask, show_progress=True)

    # -----------------------------------------------------------
    # 5. SAVING RESULTS
    # -----------------------------------------------------------
    print(f"\n[5/5] Saving Results to: {args.out}")
    save_parameter_maps(maps, affine, output_dir=args.out)
    
    # Save Metadata (for scientific reproducibility)
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": args.input,
        "protocol": {
            "n_directions": len(real_bvals),
            "b_max": float(np.max(real_bvals))
        },
        "snr_info": {
            "value": estimated_snr,
            "source": snr_source
        },
        "calibration_results": {
            "optimal_bases": optimal_bases,
            "optimal_lambda": optimal_lambda,
            "avg_calibration_error": best_params['avg_error']
        },
        "processing_time_sec": time.time() - start_time
    }
    
    meta_path = os.path.join(args.out, "pipeline_info.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"      Metadata saved to: {meta_path}")
    print("\n" + "="*50)
    print("   PROCESSING COMPLETED SUCCESSFULLY")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()