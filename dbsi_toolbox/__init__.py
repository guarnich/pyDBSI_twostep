# dbsi_toolbox/__init__.py

from .linear import DBSI_Linear
from .nonlinear import DBSI_NonLinear
from .twostep import DBSI_TwoStep  
from .common import DBSIParams
from .utils import load_dwi_data_dipy, save_parameter_maps

# Default model is the Two-Step approach
DBSIModel = DBSI_TwoStep 

__version__ = "0.1.0"
print("DBSI Toolbox v0.1.0 loaded.")