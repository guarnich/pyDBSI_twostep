# pyDBSI_twostep: Two-Step DBSI Fitting Toolbox

A Python toolbox for fitting the **Diffusion Basis Spectrum Imaging (DBSI)** model using the robust **Two-Step** approach (Linear Spectrum + Non-Linear Tensor refinement).

## 🧠 What is DBSI?

**Diffusion Basis Spectrum Imaging (DBSI)** is an advanced diffusion MRI model designed to overcome the limitations of standard Diffusion Tensor Imaging (DTI).

While DTI struggles in areas with complex pathologies (like inflammation, edema, and axonal injury co-existing) [Wang, Y. et al., 2011; Kéri, S., 2025], DBSI was developed to resolve and quantify these individual components. It achieves this by modeling the diffusion signal as a combination of:

1.  **Anisotropic Tensors:** Representing water diffusion along organized structures like axonal fibers.
2.  **A Spectrum of Isotropic Tensors:** Representing water diffusing freely in different environments [Shirani, A. et al., 2019; Cross, A.H. and Song, S.K., 2017].

### The "Two-Step" Approach

This repository implements the standard solving strategy:
1.  **Step 1 (Spectral):** Uses Non-Negative Least Squares (NNLS) with Tikhonov regularization to estimate signal fractions and fiber directions using a fixed basis set.
2.  **Step 2 (Tensor):** Uses the results from Step 1 as an initial guess for a Non-Linear Least Squares (NLLS) optimization to refine axial and radial diffusivities.

### Key Metrics

This toolbox provides maps for all DBSI parameters, allowing you to quantify distinct tissue properties:

  * **Fiber Fraction (f\_fiber):** Reflects the apparent density of axonal fibers [Shirani, A. et al., 2019; Vavasour, I.M. et al., 2022].
  * **Axial Diffusivity (D\_axial / AD):** A marker for axonal integrity; a decrease often suggests axonal injury [Lavadi, R.S. et al., 2025; Tu, T.W. et al., 2012].
  * **Radial Diffusivity (D\_radial / RD):** A marker for myelin integrity; an increase often suggests demyelination [Lavadi, R.S. et al., 2025; Shirani, A. et al., 2019].
  * **Restricted Fraction (f\_restricted):** The key isotropic component, modeling water in highly restricted environments. This metric serves as a putative marker for **cellularity** (e.g., inflammation, gliosis, or tumor cells) [Wang, Y. et al., 2011; Kéri, S., 2025].
  * **Hindered & Water Fractions (f\_hindered, f\_water):** Isotropic components representing water in less dense environments, such as vasogenic **edema** or tissue loss [Tu, T.W. et al., 2012; Shirani, A. et al., 2019].

-----

## 🚀 Installation

This package is designed to be installed in "editable" mode, allowing for easy updates and modifications.

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/guarnich/pyDBSI_twostep.git
    cd pyDBSI_twostep
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

You simply need to provide your input NIfTI files and an output directory. The toolbox will automatically estimate the Signal-to-Noise Ratio (SNR) and calibrate the model parameters (basis count and regularization lambda) for you.

```bash
python examples/run_dbsi.py \
    --input "subject/dwi/dwi_preproc.nii.gz" \
    --bval  "subject/dwi/dwi.bval" \
    --bvec  "subject/dwi/dwi.bvec" \
    --mask  "subject/dwi/brain_mask.nii.gz" \
    --out   "subject/dbsi_results"