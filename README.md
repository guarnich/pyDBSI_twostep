
# pyDBSI

Python toolbox for fitting the Diffusion Basis Spectrum Imaging (DBSI) model to diffusion-weighted MRI data.

# DBSI Fitting Examples

This folder contains two primary methods for running the DBSI model fitting using the `dbsi_toolbox`.

## 1\. Installation (Required)

Before running any examples, you must install the toolbox in "editable mode." This links the package to your Python environment, allowing you to import it.

From the **root directory** (the folder containing `setup.py`), run:

```bash
pip install -e .
```

(The `.` refers to the current directory).

-----

## Command Line Interface (Bash/CLI Style)

This is the most robust method for integrating into automated pipelines (e..g., Bash scripts, SLURM).

**Script:** `run_dbsi.py`

### How to Run

Pass the file paths as arguments directly in your terminal.

#### Command Template

```bash
python examples/run_dbsi_cli.py \
    --nii  "<path_to_your_file.nii.gz>" \
    --bval "<path_to_your_file.bval>" \
    --bvec "<path_to_your_file.bvec>" \
    --mask "<path_to_your_mask.nii.gz>" \
    --out  "<directory_for_results>" \
    --prefix "my_output_prefix"
```