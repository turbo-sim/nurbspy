import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax
jax.config.update("jax_enable_x64", True)

# Import geometry modules
from .nurbs_basis_functions import *
from .nurbs_curve import *
from .nurbs_curve_circular_arc import *

