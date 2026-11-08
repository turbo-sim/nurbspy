# nurbspy

## Description
Python library for B-Spline and NURBS curves and surfaces, with analytical derivatives, geometric analysis and support for shape optimization.
The classes and methods were inspired by the algorithms presented in [The NURBS Book](https://doi.org/10.1007/978-3-642-59223-2) and the code uses vectorized [NumPy](https://numpy.org/) operations, with optional [Numba](http://numba.pydata.org/) just-in-time compilation for faster basis-function evaluation.

`nurbspy` aims to be a simple NURBS library, not a fully fledged CAD kernel. If you need a powerful, open source CAD kernel we recommend you to check out the C++ [OpenCascade](https://www.opencascade.com/doc/occt-7.4.0/overview/html/index.html) library. If you feel that OpenCascade is too complex or you are not sure how to start using it, [this repository](https://github.com/RoberAgro/primer_open_cascade) might be useful for you!

📚 **Documentation**: [https://turbo-sim.github.io/nurbspy/](https://turbo-sim.github.io/nurbspy/)

## Capabilities

`nurbspy` represents both curves and surfaces as rational or non-rational Bézier / B-Spline / NURBS geometries, built directly from control points, weights, and knot vectors. For both, it provides:

- Evaluation of coordinates and arbitrary-order analytical derivatives
- Curvature (and, for surfaces, mean and Gaussian curvature)
- Point projection / point inversion
- Optional interactive desktop and embedded geometry viewing/editing
- Matplotlib static and diagnostic plots

Curves additionally offer the tangent, normal, and binormal unit vectors (Frenet-Serret frame), torsion, and arc-length by numerical quadrature. Surfaces additionally offer unit normal vectors, u- and v-isoparametric curves, and constructors for common special surfaces: bilinear, ruled, extruded, revolution, and Coons.

`nurbspy` also works with real and complex data types natively, so shape derivatives can be computed to machine precision with the [complex-step method](https://blogs.mathworks.com/cleve/2013/10/14/complex-step-differentiation/) instead of finite differences. This is useful for shape-optimization problems with many design variables that rely on gradient-based algorithms; `nurbspy` is, to our knowledge, the only Python NURBS package with native complex-number support.

An optional [JAX](https://github.com/google/jax) backend (`nurbspy.jax`) additionally provides automatic differentiation and JIT compilation, see [Installation](#installation).

For repeated JAX calculations at fixed knots and parameter locations,
`prepare_curve` and `prepare_surface` reuse polynomial basis tables while
control points and weights vary. `prepare_surface_projection` reuses the
initialization grid for repeated queries onto one fixed surface. See the
[JAX performance guide](https://turbo-sim.github.io/nurbspy/getting_started/jax_performance.html)
for usage, lifetimes, and tradeoffs.

See the documentation for the [theory](https://turbo-sim.github.io/nurbspy/theory/index.html) behind these capabilities (Bézier, B-Spline, NURBS, and G² continuity) and the [examples](https://turbo-sim.github.io/nurbspy/examples/index.html) gallery, with complete scripts and their numerical and graphical output.


See the [numbered tutorials](demos/README.md) for examples using automatic display
quality. Display offers Auto, Fast, and Fine; no numerical resolution is needed.

## Interactive geometry viewing and editing

The primary interactive geometry interface uses the shared `visualization/`
renderer. Desktop windows start no web server. Desktop visualization and Dash
embedding are included in the standard dependencies. Install from this checkout,
then run a curve or surface example:

```powershell
poetry install
poetry run python demos/03_viewer/01_curve.py
poetry run python demos/03_viewer/02_surface.py
```

```python
import nurbspy as nrb

curve = nrb.NurbsCurve([nrb.Point(0, 0), nrb.Point(1, 2), nrb.Point(3, 0)])
curve.show()
edited = curve.edit()
if edited is not None:
    edited.to_yaml("edited_curve.yaml")
```

View and edit use the same 3D/orthographic viewport, navigation dropdown,
ViewCube, solid-colour defaults, diagnostics, and optional STL mesh display.
Editors add direct X/Y/Z coordinates, constrained dragging, positive weights,
and Reset. Original inputs are unchanged; Apply returns a new NumPy geometry,
including for JAX inputs. YAML can restore that geometry through either backend.
Undo/redo and file dialogs are deliberately omitted.

For a host-owned Dash application, use
`GeometryView` or `GeometryEditor`. See the [guide](docs/source/visualization.md)
and [separate runnable examples](demos/03_viewer/README.md). Existing
`.plot()` methods retain their Matplotlib signatures; use `.show()`/`.edit()`
for the new geometry experience or `show(backend="matplotlib")` explicitly.

Use `nrb.Figure(geometry)` to retain camera and appearance settings, combine named
geometries, save a self-contained YAML view, and export PNG files with the same
Three.js renderer. Headless export and scripted frame generation are described
in the [figure guide](docs/source/figures.md).

## Installation

The development version requires **Python 3.12 or 3.13**. Install the latest release from [PyPI](https://pypi.org/project/nurbspy/):

```bash
pip install nurbspy
```

The default installation uses NumPy and does not require Numba. To add
optional acceleration:

```bash
pip install "nurbspy[numba]"
```

Numba is used automatically when it is available. Without it, the same
functions run with NumPy; the API and numerical algorithms are unchanged,
but evaluation can be slower. If Numba raises an import error (for example,
because it does not support the installed NumPy version), `nurbspy` warns
and uses NumPy. Installing the extra still requires versions compatible
with Numba's own NumPy constraints.

To also install the optional JAX backend and its supporting libraries:

```bash
pip install "nurbspy[jax]"
```

Use `import nurbspy as nrb` (or explicitly `import nurbspy.numpy as nrb`) for
the NumPy interface, or
`import nurbspy.jax as nrb` for the JAX interface. The default installation
does not install Numba or JAX. The JAX extra does not require Numba; both
extras can be installed with `pip install "nurbspy[numba,jax]"`.
The JAX backend currently uses the CPU.

Verify the installation:

```bash
python -c "import nurbspy; print(nurbspy.__version__)"
```

If you installed the JAX extra, check it with:

```bash
python -c "import nurbspy.jax"
```

For installing from source, contributing, and running the test suite, see
[Getting started](https://turbo-sim.github.io/nurbspy/getting_started/index.html)
in the documentation.

## Minimal working examples

### NURBS curves

`nurbspy` can be used to create Bézier, B-Spline and NURBS curves. The type of curve depends on the arguments used to initialize the curve class. As an example, the following piece of code can be used to generate a degree four Bézier curve in two dimensions:

```python
# Import packages
import numpy as np
import nurbspy as nrb
import matplotlib.pyplot as plt

# Define the array of control points
P = np.zeros((2,5))
P[:, 0] = [0.20, 0.50]
P[:, 1] = [0.40, 0.70]
P[:, 2] = [0.80, 0.60]
P[:, 3] = [0.80, 0.40]
P[:, 4] = [0.40, 0.20]

# Create and plot the Bezier curve
bezierCurve = nrb.NurbsCurve(control_points=P)
bezierCurve.plot()
plt.show()
```

If the installation was successful, you should be able to see the Bézier curve when you execute the previous code snippet.

<p align="center">
	<img src="./docs/images/curve_example.svg" height="350" width="350"/>
</p>

Check out the [documentation examples](./demos/documentation) directory to see more examples showing the capabilities of the library and how to use them.


### NURBS surfaces

Similarly, `nurbspy` can be used to create Bézier, B-Spline and NURBS surfaces. The type of surface depends on the arguments used to initialize the surface class. As an example, the following code snippet can be used to generate a simple Bézier surface of degree 3 in the u-direction and degree 2 in the v-direction:

```python
# Import packages
import numpy as np
import nurbspy as nrb
import matplotlib.pyplot as plt

# Define the array of control points
n_dim, n, m = 3, 4, 3
P = np.zeros((n_dim, n, m))

# First row
P[:, 0, 0] = [0.00, 0.00, 0.00]
P[:, 1, 0] = [1.00, 0.00, 1.00]
P[:, 2, 0] = [2.00, 0.00, 1.00]
P[:, 3, 0] = [3.00, 0.00, 0.00]

# Second row
P[:, 0, 1] = [0.00, 1.00, 1.00]
P[:, 1, 1] = [1.00, 1.00, 2.00]
P[:, 2, 1] = [2.00, 1.00, 2.00]
P[:, 3, 1] = [3.00, 1.00, 1.00]

# Third row
P[:, 0, 2] = [0.00, 2.00, 0.00]
P[:, 1, 2] = [1.00, 2.00, 1.00]
P[:, 2, 2] = [2.00, 2.00, 1.00]
P[:, 3, 2] = [3.00, 2.00, 0.00]

# Create and plot the Bezier surface
bezierSurface = nrb.NurbsSurface(control_points=P)
bezierSurface.plot(control_points=True, isocurves_u=6, isocurves_v=6)
plt.show()
```

If the installation was successful, you should be able to see the Bézier surface when you execute the previous script.

<p align="center">
	<img src="./docs/images/surface_example.svg" height="400" width="400"/>
</p>

Check out the [documentation examples](./demos/documentation) directory to see more examples showing the capabilities of the library and how to use them.

## Development

The Python package lives under `src/nurbspy`. Install the checkout in editable
mode before running local scripts or tests:

```bash
python -m pip install -e . pytest
python -m pytest
```

To test both optional backends, install `-e ".[numba,jax]"` instead and run
`python -m pytest --require-backend=numba --require-backend=jax`.
`poetry install --with dev --all-extras` also installs the checkout in editable
mode; use `poetry run python -m pytest` in that environment.

## Contact information

`nurbspy` was originally developed by [Roberto Agromayor](https://www.ntnu.edu/employees/roberto.agromayor) under the supervision of Associate Professor [Lars O. Nord](https://www.ntnu.edu/employees/lars.nord) at the [Norwegian University of Science and Technology (NTNU)](https://www.ntnu.no/) as part of his PhD on turbomachinery shape optimization, and has since been maintained and extended. Please, drop us an email to [roberto.agromayor@ntnu.no](mailto:roberto.agromayor@ntnu.no) if you have questions about the code or you have a bug to report!

<!-- research-software-metadata:start -->
## Development and citation

Roberto Agromayor is the main developer of nurbspy.

Citation metadata is provided in [CITATION.cff](CITATION.cff). For version-specific reproducibility, also record the release or commit used.
<!-- research-software-metadata:end -->

### Plotting with numerical backends

NumPy and JAX geometry share the Matplotlib plotting implementation. Plotting JAX models copies their current arrays to the host; it does not modify the model. Call plotting methods outside JIT-compiled or differentiated functions. Numerical evaluation and automatic differentiation remain on the selected backend.

### Readable control points

Use immutable `Point` values directly through the existing constructor:

```python
import nurbspy as nrb

curve = nrb.NurbsCurve(
    control_points=[nrb.Point(0, 0), nrb.Point(1, 2), nrb.Point(3, 2), nrb.Point(4, 0)],
    degree=3,
)
```

Lists and tuples of `Point` objects are converted once to backend arrays.
Weights are supplied separately. Reusing a Point does not link geometries.
Numeric lists, tuples, and arrays retain the existing coordinate-first convention
`P[coordinate, point_index]`; their layout is never guessed from shape or type.
The same `Point` is available from `nurbspy`, `nurbspy.numpy`, and `nurbspy.jax`.
Array-based construction continues to support arbitrary coordinate dimensions,
complex numbers, and JAX transformations. Point coordinates can also be complex
or scalar JAX values, including traced values inside differentiated functions.

For surfaces, supply a rectangular Point grid. The outer index advances in
`u`, the inner index in `v`, and `weights[i][j]` belongs to `points[i][j]`:

```python
surface = nrb.NurbsSurface(control_points=[
    [nrb.Point(0, 0, 0), nrb.Point(0, 1, 0)],
    [nrb.Point(1, 0, 0), nrb.Point(1, 1, 1)],
    [nrb.Point(2, 0, 0), nrb.Point(2, 1, 0)],
])
```

Internally this is still `P[:, i, j]`. Numerical arrays and nested numerical
lists retain shape `(dimension, number_u, number_v)`. A flat Point sequence is
not a surface grid. Mixed Point/numeric input, ragged rows, inconsistent Point
dimensions, and mismatched weight grids raise clear errors.

Individual locations also accept Point: circular-arc centers, bilinear-patch
corners, and revolution-axis points. Direction vectors remain numerical arrays
or sequences. The bilinear helper retains its historical argument order:
`P00=S(0,0)`, `P01=S(1,0)`, `P10=S(0,1)`, and `P11=S(1,1)`.


## Geometry files and STL export

NumPy and JAX curves/surfaces share YAML persistence. Trimesh and NetworkX are
standard dependencies, so STL export works with a normal nurbspy installation.
They are imported only when exporting; native tessellation does not load them.

```python
curve.to_yaml("curve.yaml")
restored = nrb.NurbsCurve.from_yaml("curve.yaml")
vertices, faces = surface.tessellate(nu=80, nv=30)
surface.export("surface.stl", nu=80, nv=30)
```

For multi-patch shells, use `nrb.export_stl(patches, "solid.stl", require_closed=True)`.
See [geometry I/O](docs/source/geometry_io.md) for backend selection, sampling,
closure checks, and the limits of STL precision and solid validation.


Plot and export an exact NURBS torus and a sampled ruled-surface oloid:

```bash
python demos/02_surfaces/15_stl_printing.py --quality print
```

See the [printing walkthrough](docs/source/examples/stl_printing.md) for all
sampling/export parameters, preview/fine presets, dimensions, and slicer units.
