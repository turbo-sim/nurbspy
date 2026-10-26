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
| Repeated JAX curve evaluation | {func}`nurbspy.jax.prepared.prepare_curve` |
| Repeated JAX surface evaluation | {func}`nurbspy.jax.prepared.prepare_surface` |
| Repeated projection onto a fixed JAX surface | {func}`nurbspy.jax.prepared_projection.prepare_surface_projection` |

The package overview below lists the `nurbspy.numpy` and `nurbspy.jax`
backends, their surface generators, and the shared graphics module.
The root `nurbspy` interface continues to expose the NumPy implementation.
Check backend signatures and differentiation guarantees before switching imports.
The [JAX performance guide](getting_started/jax_performance.md) explains prepared
table lifetimes, output shapes and differentiation contracts.

```{toctree}
:maxdepth: 2

api/nurbspy
```
