# Saving and loading geometry

Curves and surfaces in both backends share `to_yaml()` and `from_yaml()`:

```python
import nurbspy as nrb

curve = nrb.NurbsCurve([nrb.Point(0, 0), nrb.Point(1, 2), nrb.Point(3, 0)])
curve.to_yaml("curve.yaml")
restored = nrb.NurbsCurve.from_yaml("curve.yaml")

surface.to_yaml("surface.yaml")
restored_surface = nrb.NurbsSurface.from_yaml("surface.yaml")
```

`to_yaml(path)` returns the written `Path`. Without a destination it returns
YAML text; text streams are also supported:

```python
from io import StringIO
text = curve.to_yaml()
restored = nrb.NurbsCurve.from_yaml(StringIO(text))
```

Calling the class from `nurbspy.jax` restores JAX arrays; calling the NumPy class
restores NumPy arrays. Files are backend-independent. Save/load are host-side
operations outside JIT and differentiation; a restored JAX model can subsequently
be differentiated normally.

The versioned schema records coordinate-first control arrays, weights, degrees,
and full knot vectors. Numeric dtypes are retained where supported by the selected
backend. Complex coordinates and weights use explicit real/imaginary arrays;
no Python object tags are used. Loading uses PyYAML's safe loader and validates
schema version, shape, degree, knot order, multiplicity, and finite numeric data.
Knots must be real. Unsupported extended-precision dtypes are rejected explicitly.

Loading reconstructs the full NURBS mathematical representation. Display settings,
backend runtime state, and convenience Bezier/B-spline classification labels are
not persisted. Arrays are independent of the original geometry. This is geometry
exchange, not a pickle of an arbitrary Python object.

## Surface tessellation

```python
vertices, faces = surface.tessellate(nu=80, nv=30)
# Or sample several patches into one pair of arrays:
vertices, faces = nrb.tessellate_surfaces([surface_a, surface_b], nu=80, nv=30)
```

Vertices are NumPy arrays of shape `(N, 3)` and faces contain integer triangle
indices of shape `(M, 3)`, regardless of the numerical backend. Sampling uses
the actual knot domains and includes knot breaks, so counts may exceed `nu`
and `nv`. Triangle orientation follows the patch's $u$-cross-$v$ orientation.
Collapsed zero-area triangles are removed. Different patches retain separate
vertices: this adapter does not weld seams, orient solids, or repair topology.

Tessellation requires finite real 3D surfaces with positive finite weights.
Discontinuous interior knots are rejected; provide separate continuous patches.
Sampling resolution controls an approximation, not an error tolerance or a
watertightness guarantee. The operation transfers evaluated samples to the host
and is outside JIT/autodiff. It does not require a viewer or mesh library.
