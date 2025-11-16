# Contenuto per: examples/run_dbsi_cli.py

import sys
import argparse
import numpy as np
from time import time
import warnings

# Importa dal toolbox installato
try:
    from dbsi_toolbox.utils import load_dwi_data_dipy, save_parameter_maps
    from dbsi_toolbox.model import DBSIModel
except ImportError:
    print("ERRORE: Impossibile importare 'dbsi_toolbox'.")
    print("Assicurati di aver installato il pacchetto eseguendo 'pip install -e .' nella cartella principale.")
    sys.exit(1)

def main():
    """
    Funzione principale per eseguire il fitting DBSI da linea di comando.
    """
    
    # --- 0. Setup Argparse ---
    parser = argparse.ArgumentParser(
        description="Esegue il fitting del modello DBSI completo su dati DWI.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Argomenti richiesti
    req = parser.add_argument_group('Argomenti Richiesti')
    req.add_argument('--nii',  type=str, required=True, help="Percorso al file NIfTI 4D (.nii.gz)")
    req.add_argument('--bval', type=str, required=True, help="Percorso al file .bval")
    req.add_argument('--bvec', type=str, required=True, help="Percorso al file .bvec")
    req.add_argument('--mask',   type=str, default=None, help="Percorso alla maschera 3D. Se non fornita, viene generata automaticamente.")
    req.add_argument('--out',  type=str, required=True, help="Directory di output per le mappe NIfTI")

    # Argomenti opzionali
    opt = parser.add_argument_group('Argomenti Opzionali')
    opt.add_argument('--prefix', type=str, default='dbsi_cli', help="Prefisso per i file di output (default: 'dbsi_cli')")
    opt.add_argument('--method', type=str, default='least_squares', 
                       choices=['least_squares', 'differential_evolution'],
                       help="Algoritmo di ottimizzazione (default: 'least_squares')")
    
    args = parser.parse_args()

    print("="*70)
    print("DBSI Toolbox - Esecuzione 'Stile CLI' (Bash)")
    print("="*70)
    print(f"Input NIfTI:  {args.nii}")
    print(f"Input bval:   {args.bval}")
    print(f"Input bvec:   {args.bvec}")
    print(f"Input Maschera: {args.mask or 'Automatica'}")
    print(f"Output Dir:   {args.out}")
    print(f"Output Prefix: {args.prefix}")
    print(f"Metodo Fit:   {args.method}")
    print("="*70)

    # --- 1. Caricamento Dati ---
    print("\n[1/4] Caricamento dati...")
    try:
        data, affine, gtab, mask = load_dwi_data_dipy(
            f_nifti=args.nii,
            f_bval=args.bval,
            f_bvec=args.bvec,
            f_mask=args.mask
        )
    except Exception as e:
        print(f"ERRORE Fatale durante il caricamento dati: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Se la maschera non è stata fornita, la funzione utils la crea
    # Ma la funzione 'fit_volume' ha una sua logica interna se 'mask' è None.
    # Per coerenza, passiamo 'None' se non specificata.
    if args.mask is None:
        mask = None # Lascia che fit_volume crei la sua maschera
    
    # --- 2. Inizializzazione Modello ---
    print("\n[2/4] Inizializzazione modello DBSI completo...")
    model = DBSIModel()
    print("  ✓ Modello configurato.")

    # --- 3. Esecuzione Fitting ---
    print(f"\n[3/4] Fitting DBSI in corso (Metodo: '{args.method}')...")
    start_time = time()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        param_maps = model.fit_volume(
            volume=data, 
            bvals=gtab.bvals, 
            bvecs=gtab.bvecs, 
            mask=mask, # Passa la maschera (o None)
            method=args.method,
            show_progress=True
        )
    
    end_time = time()
    print(f"\n  ✓ Fitting completato in {end_time - start_time:.2f} secondi.")
    if w:
        print(f"  ! Sono stati generati {len(w)} warning durante il fitting.")

    # --- 4. Statistiche e Salvataggio ---
    print("\n[4/4] Statistiche e Salvataggio Mappe")
    print("-" * 70)
    
    # Per le statistiche, abbiamo bisogno di una maschera definita
    if mask is None:
        # Se la maschera era None, la peschiamo da quella generata dal modello (basata su R2>0)
        # O meglio, ricalcoliamola per sicurezza
        b0_mask = gtab.bvals < 50
        b0_volume = np.mean(data[:, :, :, b0_mask], axis=3)
        threshold = np.percentile(b0_volume[b0_volume > 0], 10)
        mask_stats = b0_volume > threshold
    else:
        mask_stats = mask
        
    valid_mask = mask_stats & (param_maps['R2'] > 0)
    
    if not np.any(valid_mask):
        print("ATTENZIONE: Nessun voxel fittato con successo. Statistiche saltate.")
    else:
        print("\n📊 STATISTICHE (Media ± Std. Dev. sui voxel validi):")
        r2_mean = np.mean(param_maps['R2'][valid_mask])
        print(f"  Qualità Fitting (R² medio): {r2_mean:.4f}")

    # Salva risultati
    save_parameter_maps(param_maps, affine, args.out, prefix=args.prefix)
    
    print("\n✓ ELABORAZIONE COMPLETATA")
    print("="*70)

if __name__ == "__main__":
    main()