# API reference

These pages are generated from the package's Python docstrings.

| Task | Interface |
| --- | --- |
| Basis functions | {func}`nurbspy.numpy.nurbs_basis_functions.compute_basis_polynomials` |
| Basis derivatives | {func}`nurbspy.numpy.nurbs_basis_functions.compute_basis_polynomials_derivatives` |
| Curves and their differential geometry | {class}`nurbspy.numpy.nurbs_curve.NurbsCurve` |
| Exact circular arcs | {class}`nurbspy.numpy.nurbs_curve_circular_arc.CircularArc` |
| Tensor-product surfaces | {class}`nurbspy.numpy.nurbs_surface.NurbsSurface` |
| Extrusions | {class}`nurbspy.numpy.nurbs_surface_extruded.NurbsSurfaceExtruded` |

The package overview below lists the `nurbspy.numpy` and `nurbspy.jax`
backends, their surface generators, and the shared graphics module.
The root `nurbspy` interface continues to expose the NumPy implementation.
Check backend signatures and differentiation guarantees before switching imports.

```{toctree}
:maxdepth: 2

api/nurbspy
```
