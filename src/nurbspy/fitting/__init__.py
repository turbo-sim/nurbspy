"""Surface least squares with optional exact-Hessian nonlinear optimization.

Importing this module does not import JAX, PySolverView or Pounce. The input
surface supplies the patch topology; fitting returns independent geometry.
"""

from ._api import SurfaceFitResult, fit_surface

__all__ = ["SurfaceFitResult", "fit_surface"]
