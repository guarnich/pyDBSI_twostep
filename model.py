# dbsi_toolbox/model.py

import numpy as np
from scipy.optimize import least_squares, differential_evolution
from dataclasses import dataclass
from typing import Tuple, Optional, List
from tqdm import tqdm
import warnings
import sys

# La tua classe DBSIParams per strutturare i risultati
@dataclass
class DBSIParams:
    """Struttura per i parametri DBSI completi"""
    # Componenti isotrope
    f_restricted: float      # Frazione ristretta (cellule, infiammazione)
    D_restricted: float      # Diffusività ristretta (~0.0-0.3 μm²/ms)
    f_hindered: float        # Frazione ostacolata (spazio extracellulare)
    D_hindered: float        # Diffusività ostacolata (~0.3-1.5 μm²/ms)
    f_water: float           # Frazione acqua libera (CSF, edema)
    D_water: float           # Diffusività acqua (~2.5-3.0 μm²/ms)
    
    # Componente anisotropa
    f_fiber: float           # Frazione fibre
    D_axial: float           # Diffusività assiale
    D_radial: float          # Diffusività radiale
    theta: float             # Angolo polare
    phi: float               # Angolo azimutale
    
    @property
    def f_iso_total(self) -> float:
        """Frazione isotropa totale"""
        return self.f_restricted + self.f_hindered + self.f_water
    
    @property
    def f_aniso_total(self) -> float:
        """Frazione anisotropa totale"""
        return self.f_fiber
    
    @property
    def fiber_FA(self) -> float:
        """Fractional Anisotropy del compartimento fibre"""
        mean_D = (self.D_axial + 2*self.D_radial) / 3
        if mean_D < 1e-10: # Evita divisione per zero
            return 0.0
        
        # Formula FA standard
        numerator = (self.D_axial - mean_D)**2 + 2*(self.D_radial - mean_D)**2
        denominator = self.D_axial**2 + 2*self.D_radial**2
        if denominator < 1e-10:
            return 0.0
        
        return np.sqrt(0.5 * numerator / denominator)
    
    @property
    def fiber_MD(self) -> float:
        """Mean Diffusivity del compartimento fibre"""
        return (self.D_axial + 2*self.D_radial) / 3
    
    @property
    def overall_FA(self) -> float:
        """FA globale pesata"""
        total_signal = self.f_fiber + self.f_iso_total
        if total_signal < 1e-10 or self.f_fiber < 1e-10:
            return 0.0
        # FA pesata dalla frazione anisotropa
        return self.fiber_FA * (self.f_fiber / total_signal)
    
    @property
    def overall_MD(self) -> float:
        """MD globale pesata da tutti i compartimenti"""
        total_f = self.f_restricted + self.f_hindered + self.f_water + self.f_fiber
        if total_f < 1e-10:
            return 0.0
        weighted_D = (self.f_restricted * self.D_restricted + 
                     self.f_hindered * self.D_hindered +
                     self.f_water * self.D_water +
                     self.f_fiber * self.fiber_MD)
        return weighted_D / total_f
    
    @property
    def cellularity_index(self) -> float:
        """Indice di cellularità (per infiammazione/tumori)"""
        iso_total = self.f_iso_total
        if iso_total < 1e-10:
            return 0.0
        return self.f_restricted / iso_total
    
    @property
    def edema_index(self) -> float:
        """Indice di edema (frazione acqua libera)"""
        iso_total = self.f_iso_total
        if iso_total < 1e-10:
            return 0.0
        return self.f_water / iso_total
    
    @property
    def fiber_density(self) -> float:
        """Densità fibre normalizzata"""
        total_signal = self.f_fiber + self.f_iso_total
        if total_signal < 1e-10:
            return 0.0
        return self.f_fiber / total_signal
    
    @property
    def axial_diffusivity(self) -> float:
        """Diffusività assiale (alias per compatibilità)"""
        return self.D_axial
    
    @property
    def radial_diffusivity(self) -> float:
        """Diffusività radiale (alias per compatibilità)"""
        return self.D_radial

# La tua classe DBSIModel per il fitting
class DBSIModel:
    """Modello DBSI completo per fitting multi-compartimentale"""
    
    def __init__(self):
        """Inizializza il modello DBSI completo"""
        self.n_params = 10  # 3 iso (f,D) + 1 aniso (f,D_ax,D_rad,theta,phi), D_water fixed
    
    def predict_signal(self, params: np.ndarray, bvals: np.ndarray, 
                      bvecs: np.ndarray, S0: float = 1.0) -> np.ndarray:
        """
        Predice il segnale DWI usando il modello DBSI completo
        
        Args:
            params: [f_res, D_res, f_hin, D_hin, f_wat, f_fib, D_ax, D_rad, theta, phi]
            bvals: Valori b (N,)
            bvecs: Vettori direzione (N, 3) normalizzati
            S0: Segnale a b=0
            
        Returns:
            Segnale predetto (N,)
        """
        f_res, D_res, f_hin, D_hin, f_wat, f_fib, D_ax, D_rad, theta, phi = params
        
        # Diffusività acqua libera fissa (tipica del CSF)
        D_water = 3.0e-3  # 3.0 μm²/ms
        
        # Normalizza frazioni
        f_total = f_res + f_hin + f_wat + f_fib
        if f_total > 1e-10:
            f_res_norm = f_res / f_total
            f_hin_norm = f_hin / f_total
            f_wat_norm = f_wat / f_total
            f_fib_norm = f_fib / f_total
        else:
            # Default se somma è zero
            f_res_norm = 0.0
            f_hin_norm = 0.0
            f_wat_norm = 0.0
            f_fib_norm = 0.0
        
        # Direzione principale delle fibre
        fiber_dir = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        
        # Tensore di diffusione anisotropo (fibre)
        D_tensor = D_rad * np.eye(3) + (D_ax - D_rad) * np.outer(fiber_dir, fiber_dir)
        
        # Calcolo segnale (vettorizzato per efficienza)
        
        # Componenti isotrope
        S_restricted = np.exp(-bvals * D_res)
        S_hindered = np.exp(-bvals * D_hin)
        S_water = np.exp(-bvals * D_water)
        
        # Componente anisotropa
        # Calcola (g * D * g) per tutti i bvecs
        g_D_g = np.einsum('ij,ji->i', np.dot(bvecs, D_tensor), bvecs.T)
        S_fiber = np.exp(-bvals * g_D_g)
        
        # Segnale totale pesato
        signal = S0 * (
            f_res_norm * S_restricted +
            f_hin_norm * S_hindered +
            f_wat_norm * S_water +
            f_fib_norm * S_fiber
        )
        
        return signal
    
    def fit_voxel(self, signal: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray,
                  method: str = 'least_squares') -> Tuple[np.ndarray, dict]:
        """
        Esegue il fitting per un singolo voxel
        
        Args:
            signal: Segnale DWI osservato (N,)
            bvals: Valori b (N,)
            bvecs: Vettori direzione (N, 3)
            method: 'least_squares' o 'differential_evolution'
            
        Returns:
            Array parametri fittati e informazioni di fitting
        """
        # Normalizza il segnale
        S0 = np.mean(signal[bvals < 50]) if np.any(bvals < 50) else signal[0]
        
        # Controlla se il voxel ha segnale valido
        if S0 < 1e-6 or np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
            return np.zeros(self.n_params), {'success': False, 'r_squared': 0.0, 'S0': 0.0}
        
        signal_norm = signal / (S0 + 1e-10)
        
        # Parametri iniziali e bounds
        # [f_res, D_res, f_hin, D_hin, f_wat, f_fib, D_ax, D_rad, theta, phi]
        params0 = [0.1, 0.0002, 0.2, 0.001, 0.1, 0.6, 0.0015, 0.0003, np.pi/4, np.pi/4]
        bounds_lower = [0.0, 0.0,     0.0, 0.0003, 0.0, 0.0, 0.0005, 0.0,     0.0,   0.0]
        bounds_upper = [1.0, 0.0003,  1.0, 0.0015, 1.0, 1.0, 0.003,  0.0015, np.pi, 2*np.pi]
        
        def objective(p):
            pred = self.predict_signal(p, bvals, bvecs, S0=1.0)
            return pred - signal_norm
        
        # Ottimizzazione
        try:
            if method == 'least_squares':
                result = least_squares(
                    objective, 
                    params0,
                    bounds=(bounds_lower, bounds_upper),
                    method='trf',
                    ftol=1e-6,
                    xtol=1e-6,
                    max_nfev=1000
                )
                params_opt = result.x
                success = result.success
                
            elif method == 'differential_evolution':
                result = differential_evolution(
                    lambda p: np.sum(objective(p)**2),
                    bounds=list(zip(bounds_lower, bounds_upper)),
                    seed=42,
                    maxiter=500,
                    polish=True
                )
                params_opt = result.x
                success = result.success
            else:
                raise ValueError(f"Metodo non supportato: {method}")
            
            # Calcola R²
            signal_pred = self.predict_signal(params_opt, bvals, bvecs, S0)
            ss_res = np.sum((signal - signal_pred)**2)
            ss_tot = np.sum((signal - np.mean(signal))**2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-10))
            
            info = {
                'success': success,
                'r_squared': r_squared,
                'S0': S0
            }
            
            return params_opt, info
            
        except Exception as e:
            warnings.warn(f"Fitting fallito: {str(e)}")
            return np.zeros(self.n_params), {'success': False, 'r_squared': 0.0, 'S0': S0}
    
    def fit_volume(self, volume: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray,
                   mask: Optional[np.ndarray] = None, method: str = 'least_squares',
                   show_progress: bool = True) -> dict:
        """
        Esegue il fitting su un volume 4D
        
        Args:
            volume: Volume DWI 4D (X, Y, Z, N)
            bvals: Valori b (N,) o (1, N)
            bvecs: Vettori direzione (3, N) o (N, 3)
            mask: Maschera 3D (X, Y, Z) - opzionale
            method: 'least_squares' o 'differential_evolution'
            show_progress: Mostra barra di progresso
            
        Returns:
            Dizionario con mappe parametriche 3D
        """
        # Verifica dimensioni
        if volume.ndim != 4:
            raise ValueError(f"Il volume deve essere 4D, ha {volume.ndim} dimensioni")
        
        X, Y, Z, N = volume.shape
        
        # Gestione bvals
        if bvals.ndim == 2:
            bvals = bvals.flatten()
        if len(bvals) != N:
            raise ValueError(f"bvals ha {len(bvals)} elementi, ma il volume ha {N} volumi")
        
        # Gestione bvecs: converti da (3, N) a (N, 3) se necessario
        if bvecs.shape == (3, N):
            bvecs = bvecs.T
        elif bvecs.shape != (N, 3):
            raise ValueError(f"bvecs deve essere (3, {N}) o ({N}, 3), è {bvecs.shape}")
        
        # Normalizza i bvecs
        bvecs_norm = bvecs / (np.linalg.norm(bvecs, axis=1, keepdims=True) + 1e-10)
        
        # Crea o valida la maschera
        if mask is None:
            b0_idx = np.argmin(bvals)
            b0_volume = volume[:, :, :, b0_idx]
            threshold = np.percentile(b0_volume[b0_volume > 0], 10)
            mask = b0_volume > threshold
        
        if mask.shape != (X, Y, Z):
            raise ValueError(f"La maschera deve avere dimensioni ({X}, {Y}, {Z}), ha {mask.shape}")
        
        # Inizializza TUTTE le mappe parametriche possibili
        param_maps = {
            # Frazioni compartimenti specifici
            'f_restricted': np.zeros((X, Y, Z)),
            'D_restricted': np.zeros((X, Y, Z)),
            'f_hindered': np.zeros((X, Y, Z)),
            'D_hindered': np.zeros((X, Y, Z)),
            'f_water': np.zeros((X, Y, Z)),
            'f_fiber': np.zeros((X, Y, Z)),
            'D_axial': np.zeros((X, Y, Z)),
            'D_radial': np.zeros((X, Y, Z)),
            
            # Frazioni aggregate
            'f_iso': np.zeros((X, Y, Z)),      # Somma tutte le frazioni isotrope
            'f_aniso': np.zeros((X, Y, Z)),    # Frazione anisotropa (alias di f_fiber)
            
            # Orientamento fibre
            'theta': np.zeros((X, Y, Z)),
            'phi': np.zeros((X, Y, Z)),
            
            # Metriche anisotropia e diffusività
            'fiber_FA': np.zeros((X, Y, Z)),         # FA del compartimento fibre
            'fiber_MD': np.zeros((X, Y, Z)),         # MD del compartimento fibre
            'overall_FA': np.zeros((X, Y, Z)),       # FA globale pesata
            'overall_MD': np.zeros((X, Y, Z)),       # MD globale pesata
            'FA': np.zeros((X, Y, Z)),               # Alias di overall_FA
            'MD': np.zeros((X, Y, Z)),               # Alias di overall_MD
            'AD': np.zeros((X, Y, Z)),               # Axial Diffusivity (D_axial)
            'RD': np.zeros((X, Y, Z)),               # Radial Diffusivity (D_radial)
            
            # Indici patologici
            'cellularity_index': np.zeros((X, Y, Z)),  # Infiammazione
            'edema_index': np.zeros((X, Y, Z)),        # Edema
            'fiber_density': np.zeros((X, Y, Z)),      # Densità fibre
            
            # Qualità fitting
            'R2': np.zeros((X, Y, Z)),
            'S0': np.zeros((X, Y, Z))
        }
        
        # Conta voxel da processare
        n_voxels = np.sum(mask)
        
        # Fitting per ogni voxel con barra di progresso
        if show_progress:
            progress_bar = tqdm(total=n_voxels, desc="Fitting DBSI", 
                              position=0, leave=True, file=sys.stdout,
                              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        else:
            progress_bar = None
        
        voxel_count = 0
        for x in range(X):
            for y in range(Y):
                for z in range(Z):
                    if not bool(mask[x, y, z]):
                        continue
                    
                    signal = volume[x, y, z, :]
                    params, info = self.fit_voxel(signal, bvals, bvecs_norm, method)
                    
                    if info['success']:
                        # Parametri base
                        param_maps['f_restricted'][x, y, z] = params[0]
                        param_maps['D_restricted'][x, y, z] = params[1]
                        param_maps['f_hindered'][x, y, z] = params[2]
                        param_maps['D_hindered'][x, y, z] = params[3]
                        param_maps['f_water'][x, y, z] = params[4]
                        param_maps['f_fiber'][x, y, z] = params[5]
                        param_maps['D_axial'][x, y, z] = params[6]
                        param_maps['D_radial'][x, y, z] = params[7]
                        param_maps['theta'][x, y, z] = params[8]
                        param_maps['phi'][x, y, z] = params[9]
                        
                        # Crea oggetto DBSIParams per calcolare metriche derivate
                        dbsi_params = DBSIParams(
                            f_restricted=params[0], 
                            D_restricted=params[1], 
                            f_hindered=params[2], 
                            D_hindered=params[3],
                            f_water=params[4], 
                            D_water=3.0e-3,
                            f_fiber=params[5], 
                            D_axial=params[6], 
                            D_radial=params[7],
                            theta=params[8], 
                            phi=params[9]
                        )
                        
                        # Frazioni aggregate
                        param_maps['f_iso'][x, y, z] = dbsi_params.f_iso_total
                        param_maps['f_aniso'][x, y, z] = dbsi_params.f_aniso_total
                        
                        # Metriche anisotropia e diffusività
                        param_maps['fiber_FA'][x, y, z] = dbsi_params.fiber_FA
                        param_maps['fiber_MD'][x, y, z] = dbsi_params.fiber_MD
                        param_maps['overall_FA'][x, y, z] = dbsi_params.overall_FA
                        param_maps['overall_MD'][x, y, z] = dbsi_params.overall_MD
                        param_maps['FA'][x, y, z] = dbsi_params.overall_FA  # Alias
                        param_maps['MD'][x, y, z] = dbsi_params.overall_MD  # Alias
                        param_maps['AD'][x, y, z] = dbsi_params.axial_diffusivity
                        param_maps['RD'][x, y, z] = dbsi_params.radial_diffusivity
                        
                        # Indici patologici
                        param_maps['cellularity_index'][x, y, z] = dbsi_params.cellularity_index
                        param_maps['edema_index'][x, y, z] = dbsi_params.edema_index
                        param_maps['fiber_density'][x, y, z] = dbsi_params.fiber_density
                        
                        # Qualità
                        param_maps['R2'][x, y, z] = info['r_squared']
                        param_maps['S0'][x, y, z] = info['S0']
                    
                    voxel_count += 1
                    if progress_bar is not None:
                        progress_bar.update(1)
        
        if progress_bar is not None:
            progress_bar.close()
        
        return param_maps