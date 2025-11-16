# pyDBSI
Python toolbox for fitting the Diffusion Basis Spectrum Imaging (DBSI) model to diffusion-weighted MRI data.

# Esempi di Fitting DBSI

Questa cartella contiene due modi principali per eseguire il fitting del modello DBSI utilizzando il `dbsi_toolbox`.

## 1. Installazione (Necessaria)

Prima di eseguire qualsiasi esempio, devi installare il toolbox in "modalità sviluppo". Questo collega il pacchetto al tuo ambiente Python, permettendoti di importarlo.

Dalla **cartella principale** (la directory che contiene `setup.py`), esegui:

```bash
pip install -e .
```

(Il `.` si riferisce alla cartella corrente).

---

## Script da Linea di Comando (Stile Bash/CLI)

Questo è il metodo più robusto per l'integrazione in pipeline automatiche (es. script Bash, SLURM).

**Script:** `run_dbsi_cli.py`

### Come Eseguirlo

Passa i percorsi dei file come argomenti direttamente nel terminale.

#### Template del Comando

```bash
python examples/run_dbsi_cli.py \
    --nii  "<percorso_al_tuo_file.nii.gz>" \
    --bval "<percorso_al_tuo_file.bval>" \
    --bvec "<percorso_al_tuo_file.bvec>" \
    --mask "<percorso_alla_tua_maschera.nii.gz>" \
    --out  "<directory_per_i_risultati>" \
    --prefix "mio_prefisso_output"
```
