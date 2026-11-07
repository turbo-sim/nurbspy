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
and is outside JIT/autodiff. It does not load a viewer or mesh library.

## STL export

Trimesh and NetworkX are standard dependencies installed with nurbspy. They
handle mesh processing and writing; no extra installation option is needed.
Ordinary geometry use and `tessellate()` do not import these packages.

```python
surface.export("surface.stl", nu=80, nv=30)

# Export a collection of patches forming one closed shell:
nrb.export_stl(patches, "solid.stl", nu=50, nv=50, require_closed=True)
```

NumPy and JAX surfaces share the same `export()` method. It returns the written
`Path` and supports `.stl` files only. Curves have no surface area and cannot
be exported as STL; use YAML to preserve their native geometry instead.

The exporter normalizes coordinates for processing, delegates decimal-rounded
seam merging, duplicate/degenerate-face cleanup, and winding repair to Trimesh,
then writes a binary STL. `weld_tolerance` is an approximate absolute rounding
scale; its default is $10^{-10}$ times the largest extent. It must be at least
$10^{-15}$ times the extent and smaller than the extent. This rounding scale is
not a guaranteed Euclidean merge radius. Boundary samples must match; different
patch parameterizations are not automatically stitched. No openings are capped
or holes filled.

By default, open surfaces are permitted. `require_closed=True` additionally
checks edge closure, consistent winding, positive volume, and a single connected
shell. Disconnected solids and nested cavity shells are deliberately rejected
in this mode; export separate solids separately. These checks do not prove
absence of geometric self-intersections or general suitability for printing.

STL has no units and stores float32 coordinates. Range overflow and precision
collapse are rejected before the destination is written. Very small coordinate
scales that the mesh library cannot resolve into unit normals are also rejected,
rather than writing zero normals. Recenter or rescale
geometry when coordinate offsets overwhelm feature sizes. Existing files are
not replaced when geometry validation fails. Sampling remains an approximation,
with no tolerance-controlled geometric accuracy guarantee.

The dependency-backed behavior was verified against Trimesh 5.2.0 and NetworkX
3.7. See [Trimesh processing](https://trimesh.org/trimesh.html) and
[its STL writer](https://trimesh.org/trimesh.exchange.stl.html).

For complete plotted torus/oloid examples, presets, and parameter-selection
guidance, see [the 3D-printing walkthrough](examples/stl_printing.md).
