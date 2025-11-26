# dbsi_toolbox/twostep.py

import numpy as np
from .base import BaseDBSI
from .spectrum_basis import DBSI_BasisSpectrum
from .nlls_tensor_fit import DBSI_TensorFit
from .common import DBSIParams

class DBSI_TwoStep(BaseDBSI):
    """
    Implementazione Standard del DBSI (Two-Step).
    1. Linear Basis Spectrum (NNLS) -> Stima frazioni e direzioni
    2. Non-Linear Tensor Fit (NLLS) -> Raffina diffusività
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
        Setup iniziale prima di lanciare il loop della classe base.
        """
        print("[DBSI] Initializing Standard Two-Step Fit...")
        
        # Standardizzazione input
        flat_bvals = np.array(bvals).flatten()
        N = len(flat_bvals)
        
        if bvecs.shape == (3, N):
            current_bvecs = bvecs.T
        else:
            current_bvecs = bvecs
            
        # Pre-calcolo della matrice di design (solo una volta per tutto il volume)
        print(f"[DBSI] Pre-calculating Design Matrix ({self.spectrum_model.n_iso_bases} isotropic bases)...")
        self.spectrum_model.design_matrix = self.spectrum_model._build_design_matrix(flat_bvals, current_bvecs)
        
        # Condividi i vettori gradienti con i sottomodelli
        self.spectrum_model.current_bvecs = current_bvecs
        self.fitting_model.current_bvecs = current_bvecs
        
        # Lancia il fit voxel-by-voxel gestito da BaseDBSI
        return super().fit_volume(volume, bvals, bvecs, **kwargs)

    def fit_voxel(self, signal: np.ndarray, bvals: np.ndarray) -> DBSIParams:
        # --- STEP 1: Basis Spectrum (NNLS) ---
        # Trova rapidamente le frazioni e le direzioni principali
        spectrum_result = self.spectrum_model.fit_voxel(signal, bvals)
        
        # Se il fit lineare non trova nulla o fallisce, restituisci quello che hai
        if spectrum_result.f_fiber == 0 and spectrum_result.f_iso_total == 0:
            return spectrum_result
            
        # --- STEP 2: Tensor Fit (NLLS) ---
        # Usa il risultato dello spettro come "initial guess" per raffinare i parametri
        # Nota: Se vuoi velocizzare, puoi commentare questo step e restituire spectrum_result
        try:
            final_result = self.fitting_model.fit_voxel(signal, bvals, initial_guess=spectrum_result)
            return final_result
        except Exception:
            # Fallback al risultato lineare se il non-lineare fallisce
            return spectrum_result