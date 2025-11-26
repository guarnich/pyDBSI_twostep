# dbsi_toolbox/nlls_tensor_fit_dl.py

import numpy as np
from scipy.optimize import least_squares
from .nlls_tensor_fit import DBSI_TensorFit
from .common import DBSIParams
from typing import Optional, List

class DBSI_TensorFit_DL(DBSI_TensorFit):
    """
    Estensione del fit non lineare che include una regolarizzazione basata su Deep Learning.
    
    La funzione di costo diventa:
    Cost = || S_pred - S_mis ||^2 + alpha * || Frazioni_pred - Frazioni_DL ||^2
    """
    def __init__(self, alpha_reg: float = 0.5):
        """
        Args:
            alpha_reg: Peso della regolarizzazione. 
                       0.5 significa che l'errore sulle frazioni pesa metà dell'errore sul segnale.
        """
        super().__init__()
        self.alpha_reg = alpha_reg

    def fit_voxel(self, 
                  signal: np.ndarray, 
                  bvals: np.ndarray, 
                  initial_guess: Optional[DBSIParams] = None,
                  dl_priors: Optional[np.ndarray] = None) -> DBSIParams:
        """
        Args:
            dl_priors: Array numpy [f_res, f_hin, f_wat, f_fiber] predetti dalla rete.
        """
        
        # Normalizzazione segnale
        if np.any(bvals < 50):
            S0 = np.mean(signal[bvals < 50])
        else:
            S0 = signal[0]
            
        if S0 <= 1e-6 or not np.all(np.isfinite(signal)): 
            return self._get_empty_params()
        y = signal / S0
        
        # --- 1. Initial Guess (p0) ---
        # Se abbiamo i priors DL, usiamoli per migliorare la stima iniziale delle frazioni
        if initial_guess and initial_guess.f_fiber > 0:
            # Usa output step lineare se buono
            base_guess = initial_guess
            theta, phi = self._vector_to_angles(base_guess.fiber_dir)
        else:
            # Fallback generico
            theta, phi = (0.0, 0.0) # Default
            
        # Se DL priors disponibili, sovrascriviamo le frazioni iniziali con quelle DL (spesso più accurate)
        if dl_priors is not None:
            # dl_priors = [f_res, f_hin, f_wat, f_fib]
            f_res_0, f_hin_0, f_wat_0, f_fib_0 = dl_priors
        elif initial_guess:
            f_tot = initial_guess.f_iso_total + initial_guess.f_fiber + 1e-9
            f_res_0 = initial_guess.f_restricted / f_tot
            f_hin_0 = initial_guess.f_hindered / f_tot
            f_wat_0 = initial_guess.f_water / f_tot
            f_fib_0 = initial_guess.f_fiber / f_tot
        else:
            f_res_0, f_hin_0, f_wat_0, f_fib_0 = 0.1, 0.2, 0.1, 0.6

        # Vettore parametri: [f_res, D_res, f_hin, D_hin, f_wat, f_fib, D_ax, D_rad, theta, phi]
        p0 = [
            f_res_0, 0.0002,    # Restricted
            f_hin_0, 0.0010,    # Hindered
            f_wat_0, f_fib_0,   # Water, Fiber
            1.5e-3, 0.3e-3,     # D_ax, D_rad
            theta, phi
        ]

        # --- 2. Ottimizzazione ---
        # Limiti (Bounds)
        bounds_lower = [0.0, 0.0,     0.0, 0.0003, 0.0, 0.0, 0.0005, 0.0,     0.0,   -np.pi]
        bounds_upper = [1.0, 0.0003,  1.0, 0.0020, 1.0, 1.0, 0.0030, 0.0015,  np.pi,  2*np.pi]

        # Funzione Obiettivo Regolarizzata
        # Scipy least_squares minimizza la somma dei quadrati del vettore ritornato.
        # Per aggiungere un termine di regolarizzazione lambda * (x - target)^2, 
        # dobbiamo appendere al vettore dei residui il valore: sqrt(lambda) * (x - target).
        
        def objective_with_priors(p):
            # 1. Residui del segnale (Fisica)
            # Dimensione: (N_misure,)
            signal_residuals = self._predict_signal(p, bvals, self.current_bvecs) - y
            
            if dl_priors is None:
                return signal_residuals
            
            # 2. Residui di Regolarizzazione (Machine Learning)
            # p[0]=f_res, p[2]=f_hin, p[4]=f_wat, p[5]=f_fib
            # Normalizziamo le frazioni correnti per confrontarle con DL (che è normalizzato softmax)
            f_sum = p[0] + p[2] + p[4] + p[5] + 1e-9
            
            # dl_priors = [target_res, target_hin, target_wat, target_fib]
            reg_residuals = [
                (p[0]/f_sum - dl_priors[0]), # Errore Restricted
                (p[2]/f_sum - dl_priors[1]), # Errore Hindered
                (p[4]/f_sum - dl_priors[2]), # Errore Water
                (p[5]/f_sum - dl_priors[3])  # Errore Fiber
            ]
            
            # Moltiplichiamo per il peso alpha (radice quadrata perché poi viene elevato al quadrato)
            reg_residuals = np.array(reg_residuals) * np.sqrt(self.alpha_reg)
            
            # Concateniamo tutto in un unico vettore che least_squares minimizzerà
            return np.concatenate([signal_residuals, reg_residuals])

        try:
            res = least_squares(objective_with_priors, p0, bounds=(bounds_lower, bounds_upper), method='trf')
            p = res.x
            
            # Ricostruzione parametri (identica alla classe base)
            fiber_dir = np.array([
                np.sin(p[8]) * np.cos(p[9]),
                np.sin(p[8]) * np.sin(p[9]),
                np.cos(p[8])
            ])
            
            f_total = p[0] + p[2] + p[4] + p[5] + 1e-10
            
            # Nota: R-squared va calcolato solo sui residui del segnale, non sulla regolarizzazione
            signal_pred = self._predict_signal(p, bvals, self.current_bvecs)
            ss_res = np.sum((signal_pred - y)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1 - (ss_res/ss_tot) if ss_tot > 0 else 0
            
            return DBSIParams(
                f_restricted=p[0]/f_total,
                f_hindered=p[2]/f_total,
                f_water=p[4]/f_total,
                f_fiber=p[5]/f_total,
                fiber_dir=fiber_dir,
                axial_diffusivity=p[6],
                radial_diffusivity=p[7],
                r_squared=r2
            )
        except Exception:
            return self._get_empty_params()