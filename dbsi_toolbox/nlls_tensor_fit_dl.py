# dbsi_toolbox/nlls_tensor_fit_dl.py

import numpy as np
from scipy.optimize import least_squares
from .nlls_tensor_fit import DBSI_TensorFit
from .common import DBSIParams
from typing import Optional

class DBSI_TensorFit_DL(DBSI_TensorFit):
    """
    Esegue il fit non lineare dei tensori usando le predizioni DL come:
    1. Initial Guess (punto di partenza intelligente)
    2. Regolarizzazione (penalità se ci si allontana troppo dalla rete)
    """
    def __init__(self, alpha_reg: float = 0.5):
        super().__init__()
        self.alpha_reg = alpha_reg

    def fit_voxel(self, signal: np.ndarray, bvals: np.ndarray, 
                  initial_guess: Optional[DBSIParams] = None,
                  dl_priors: Optional[np.ndarray] = None) -> DBSIParams:
        
        # Normalizzazione
        if np.any(bvals < 50):
            S0 = np.mean(signal[bvals < 50])
        else:
            S0 = signal[0]
        
        if S0 <= 1e-6: return self._get_empty_params()
        y = signal / S0

        # --- Initial Guess dai Priors DL ---
        if dl_priors is not None:
            # dl_priors = [f_res, f_hin, f_wat, f_fib]
            p0_fractions = dl_priors
        else:
            p0_fractions = [0.1, 0.2, 0.1, 0.6]

        # [f_res, D_res, f_hin, D_hin, f_wat, f_fib, D_ax, D_rad, theta, phi]
        p0 = [
            p0_fractions[0], 0.0001, # Restricted
            p0_fractions[1], 0.0015, # Hindered
            p0_fractions[2], p0_fractions[3], # Water, Fiber
            1.5e-3, 0.3e-3,          # D_ax, D_rad
            0.0, 0.0                 # Angoli (iniziamo neutri o da fit lineare)
        ]
        
        # Bounds
        lower = [0, 0,      0, 0.0003, 0, 0, 0.0005, 0.0,    -np.pi, -np.pi]
        upper = [1, 0.0003, 1, 0.0025, 1, 1, 0.0030, 0.0010, np.pi,   np.pi]

        # Funzione di costo ibrida
        def objective(p):
            # Errore Fit Segnale (Fisica)
            y_pred = self._predict_signal(p, bvals, self.current_bvecs)
            res_signal = y_pred - y
            
            if dl_priors is None:
                return res_signal
                
            # Errore Regolarizzazione (Machine Learning)
            # Penalizziamo la differenza tra le frazioni trovate e quelle predette dalla rete
            # Frazioni attuali normalizzate
            f_sum = p[0] + p[2] + p[4] + p[5] + 1e-9
            curr_fracs = np.array([p[0], p[2], p[4], p[5]]) / f_sum
            
            res_reg = (curr_fracs - dl_priors) * np.sqrt(self.alpha_reg)
            
            return np.concatenate([res_signal, res_reg])

        try:
            res = least_squares(objective, p0, bounds=(lower, upper), method='trf')
            p = res.x
            
            # Calcolo output (come nel fit standard)
            f_sum = p[0] + p[2] + p[4] + p[5] + 1e-9
            
            # Calcola R2 solo sul segnale (senza regolarizzazione)
            y_final = self._predict_signal(p, bvals, self.current_bvecs)
            ss_res = np.sum((y - y_final)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1 - (ss_res/ss_tot) if ss_tot > 0 else 0

            fiber_dir = np.array([
                np.sin(p[8]) * np.cos(p[9]),
                np.sin(p[8]) * np.sin(p[9]),
                np.cos(p[8])
            ])

            return DBSIParams(
                f_restricted=p[0]/f_sum,
                f_hindered=p[2]/f_sum,
                f_water=p[4]/f_sum,
                f_fiber=p[5]/f_sum,
                fiber_dir=fiber_dir,
                axial_diffusivity=p[6],
                radial_diffusivity=p[7],
                r_squared=r2
            )

        except Exception:
            return self._get_empty_params()