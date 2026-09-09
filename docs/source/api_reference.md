# API reference

These pages are generated from the package's Python docstrings.

| Task | Interface |
| --- | --- |
| Basis functions | {func}`nurbspy.nurbs_basis_functions.compute_basis_polynomials` |
| Basis derivatives | {func}`nurbspy.nurbs_basis_functions.compute_basis_polynomials_derivatives` |
| Curves and their differential geometry | {class}`nurbspy.nurbs_curve.NurbsCurve` |
| Exact circular arcs | {class}`nurbspy.nurbs_curve_circular_arc.CircularArc` |
| Tensor-product surfaces | {class}`nurbspy.nurbs_surface.NurbsSurface` |
| Extrusions | {class}`nurbspy.nurbs_surface_extruded.NurbsSurfaceExtruded` |

The package overview below also lists the bilinear, ruled, revolution, Coons,
and graphics modules, and the `nurbspy.jax` subpackage. The JAX API is separate
from the NumPy API: check its signatures before switching imports.

```{toctree}
:maxdepth: 2

api/nurbspy
```
