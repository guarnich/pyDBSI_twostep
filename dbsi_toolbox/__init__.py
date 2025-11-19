# dbsi_toolbox/__init__.py

from .spectrum_basis import DBSI_BasisSpectrum
from .nlls_tensor_fit import DBSI_TensorFit
from .twostep import DBSI_TwoStep  
from .common import DBSIParams
from .utils import load_dwi_data_dipy, save_parameter_maps

# Default model is the Two-Step approach
DBSIModel = DBSI_TwoStep 

__version__ = "0.2.0"
print("DBSI Toolbox v0.2.0 loaded.")