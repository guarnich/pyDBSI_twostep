
````markdown
# pyDBSI: Python DBSI Fitting Toolbox

A Python toolbox for fitting the Diffusion Basis Spectrum Imaging (DBSI) model to diffusion-weighted MRI data.

## 🧠 What is DBSI?

**Diffusion Basis Spectrum Imaging (DBSI)** is an advanced diffusion MRI model designed to overcome the limitations of standard Diffusion Tensor Imaging (DTI).

While DTI struggles in areas with complex pathologies (like inflammation, edema, and axonal injury co-existing) [Wang, Y. et al., 2011; Kéri, S., 2025], DBSI was developed to resolve and quantify these individual components. It achieves this by modeling the diffusion signal as a combination of:

1.  **Anisotropic Tensors:** Representing water diffusion along organized structures like axonal fibers.
2.  **A Spectrum of Isotropic Tensors:** Representing water diffusing freely in different environments [Shirani, A. et al., 2019; Cross, A.H. and Song, S.K., 2017].

### Key Metrics

This toolbox provides maps for all DBSI parameters, allowing you to quantify distinct tissue properties:

  * **Fiber Fraction (f\_fiber):** Reflects the apparent density of axonal fibers [Shirani, A. et al., 2019; Vavasour, I.M. et al., 2022].
  * **Axial Diffusivity (D\_axial / AD):** A marker for axonal integrity; a decrease often suggests axonal injury [Lavadi, R.S. et al., 2025; Tu, T.W. et al., 2012].
  * **Radial Diffusivity (D\_radial / RD):** A marker for myelin integrity; an increase often suggests demyelination [Lavadi, R.S. et al., 2025; Shirani, A. et al., 2019].
  * **Restricted Fraction (f\_restricted):** The key isotropic component, modeling water in highly restricted environments. This metric serves as a putative marker for **cellularity** (e.g., inflammation, gliosis, or tumor cells) [Wang, Y. et al., 2011; Kéri, S., 2025].
  * **Hindered & Water Fractions (f\_hindered, f\_water):** Isotropic components representing water in less dense environments, such as vasogenic **edema** or tissue loss [Tu, T.W. et al., 2012; Shirani, A. et al., 2019].

This project provides a simple, open-source Python implementation to fit this powerful model to your data.

-----

## 🚀 Installation

This package is designed to be installed in "editable" mode, which allows you to use it as a toolbox while continuing to develop it.

1.  **Clone the Repository:**

    ```bash
    git clone [https://github.com/guarnich/pyDBSI.git](https://github.com/guarnich/pyDBSI.git)
    cd dbsi-fitting-toolbox
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Install the Toolbox:**

    ```bash
    pip install -e .
    ```

    (The `.` refers to the current directory). This makes the `dbsi_toolbox` importable from anywhere in your Python environment.

-----

## ⚡ Quickstart: How to Run

This toolbox is run from the command line using the `examples/run_dbsi.py` script. This script handles the complete pipeline: **Data Loading -> Automatic SNR Estimation -> Hyperparameter Calibration -> Two-Step Fitting**.

### Basic Usage

You simply need to provide your input NIfTI files and an output directory. The toolbox will automatically estimate the Signal-to-Noise Ratio (SNR) and calibrate the model parameters for you.

```bash
python examples/run_dbsi.py \
    --input "subject/dwi/dwi_preproc.nii.gz" \
    --bval  "subject/dwi/dwi.bval" \
    --bvec  "subject/dwi/dwi.bvec" \
    --mask  "subject/dwi/brain_mask.nii.gz" \
    --out   "subject/dbsi_results"
````

### Advanced Usage

If you prefer more control, you can manually specify the SNR (skipping the auto-estimation step) or adjust the number of Monte Carlo iterations used for calibration.

```bash
python examples/run_dbsi.py \
    --input "data.nii.gz" \
    --bval "data.bval" \
    --bvec "data.bvec" \
    --mask "mask.nii.gz" \
    --out  "results/" \
    --snr 35.0 \
    --mc_iter 500
```

This command will fit the DBSI model to every voxel inside the mask and save parameter maps (e.g., `dbsi_fiber_fraction.nii.gz`, `dbsi_restricted_fraction.nii.gz`, etc.) along with a `pipeline_info.json` metadata file in the output folder.

-----

## 🛠️ Command-Line Options

You can view all available command-line options and their descriptions by running the script with the `--help` flag:

```bash
python examples/run_dbsi.py --help
```

### Full List of Arguments

| Argument | Flag | Required | Description |
| :--- | :--- | :---: | :--- |
| **Input File** | `--input` | ✅ | Path to the 4D DWI NIfTI file (`.nii` or `.nii.gz`). |
| **B-Values** | `--bval` | ✅ | Path to the `.bval` file containing diffusion gradient strengths. |
| **B-Vectors** | `--bvec` | ✅ | Path to the `.bvec` file containing diffusion gradient directions. |
| **Brain Mask** | `--mask` | ✅ | Path to the 3D binary brain mask NIfTI file. **Mandatory.** |
| **Output Dir** | `--out` | ✅ | Directory where the resulting parameter maps and metadata will be saved. |
| **SNR** | `--snr` | ❌ | Manually specify the Signal-to-Noise Ratio (float). If omitted, SNR is **automatically estimated** from the data. |
| **MC Iterations** | `--mc_iter` | ❌ | Number of Monte Carlo iterations to perform during the hyperparameter calibration step. Default: `200`. |

```
```