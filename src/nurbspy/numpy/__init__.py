"""NumPy curves and surfaces, with optional Numba basis-function acceleration.

These are the same geometry objects exported by ``nurbspy``. Importing this
backend does not require or initialize JAX.
"""

from .nurbs_basis_functions import *
from .nurbs_curve import *
from .nurbs_curve_circular_arc import *
from .nurbs_surface import *
from .nurbs_surface_bilinear import *
from .nurbs_surface_ruled import *
from .nurbs_surface_extruded import *
from .nurbs_surface_revolution import *
from .nurbs_surface_coons import *
