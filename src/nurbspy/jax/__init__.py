import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax
jax.config.update("jax_enable_x64", True)

# Import geometry modules
from .nurbs_basis_functions import *
from .nurbs_curve import *
from .nurbs_curve_circular_arc import *
from .nurbs_surface import *
from .nurbs_surface_bilinear import *
from .nurbs_surface_ruled import *
from .nurbs_surface_extruded import *
from .nurbs_surface_revolution import *
from .nurbs_surface_coons import *
from .prepared import *
from .prepared_projection import *
