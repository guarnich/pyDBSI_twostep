# dbsi_toolbox/twostep.py

import numpy as np
from typing import Dict, Optional
from .base import BaseDBSI
from .spectrum_basis import DBSI_BasisSpectrum
from .nlls_tensor_fit import DBSI_TensorFit
from .common import DBSIParams

class DBSI_TwoStep(BaseDBSI):
    """
    Orchestrator class that implements the DBSI Two-Step approach:
    1. Basis Spectrum (NNLS) -> Find active compartments.
    2. Tensor Fit (NLLS) -> Refine parameters.
    """
    
    def __init__(self, 
                 iso_diffusivity_range=(0.0, 3.0e-3),
                 n_iso_bases=20,
                 reg_lambda=0.01,
                 filter_threshold=0.01,
                 axial_diff_basis=1.5e-3,
                 radial_diff_basis=0.3e-3):
        
        # Inizializza i due modelli interni
        self.spectrum_model = DBSI_BasisSpectrum(
            iso_diffusivity_range=iso_diffusivity_range,
            n_iso_bases=n_iso_bases,
            axial_diff_basis=axial_diff_basis,
            radial_diff_basis=radial_diff_basis,
            reg_lambda=reg_lambda,
            filter_threshold=filter_threshold
        )
        
        self.fitting_model = DBSI_TensorFit()
        
    def fit_volume(self, volume, bvals, bvecs, **kwargs):
        """
        Overrides fit_volume to setup the Design Matrix first.
        """
        # --- 1. PREPARAZIONE DATI LOCALE ---
        # Necessario perché dobbiamo usare questi dati PRIMA di chiamare super()
        flat_bvals = np.array(bvals).flatten()
        N = len(flat_bvals)
        
        # Standardizziamo i bvecs (N, 3)
        if bvecs.shape == (3, N):
            current_bvecs = bvecs.T
        else:
            current_bvecs = bvecs
            
        # --- 2. COSTRUZIONE MATRICE (Step Spectrum) ---
        print("Step 1/2: Pre-calculating Basis Spectrum Matrix...", end="")
        
        # Costruiamo la matrice usando i dati appena preparati
        self.spectrum_model.design_matrix = self.spectrum_model._build_design_matrix(flat_bvals, current_bvecs)
        
        # Condividiamo i vettori correnti con i sottomodelli
        self.spectrum_model.current_bvecs = current_bvecs
        self.fitting_model.current_bvecs = current_bvecs
        print(" Done.")
        
        print("Step 2/2: Running Two-Step DBSI...")
        
        # --- 3. AVVIO CICLO (Base Class) ---
        # Passiamo tutto alla classe base che gestisce il loop tqdm
        return super().fit_volume(volume, bvals, bvecs, **kwargs)

    def fit_voxel(self, signal: np.ndarray, bvals: np.ndarray) -> DBSIParams:
        # --- STEP 1: Basis Spectrum (NNLS) ---
        # Trova rapidamente le frazioni e le direzioni principali
        spectrum_result = self.spectrum_model.fit_voxel(signal, bvals)
        
        # Se il fit lineare fallisce (es. voxel vuoto), saltiamo il non-lineare
        if spectrum_result.f_fiber == 0 and spectrum_result.f_iso_total == 0:
            return spectrum_result
            
        # --- STEP 2: Tensor Fit (NLLS) ---
        # Usa il risultato dello spettro come "initial guess" per raffinare i parametri
        final_result = self.fitting_model.fit_voxel(signal, bvals, initial_guess=spectrum_result)
        
        return final_result