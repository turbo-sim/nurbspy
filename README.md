# nurbspy

## Description
`nurbspy` is a Python package for Non-Uniform Rational Basis Spline (NURBS) curves and surfaces.
The classes and methods were inspired by the algorithms presented in [The NURBS Book](https://doi.org/10.1007/978-3-642-59223-2) and the code was implemented using vectorized [Numpy](https://numpy.org/) functions and [Numba's](http://numba.pydata.org/) just-in-time compilation decorators to achieve C-like speed.

`nurbspy` aims to be a simple NURBS library, not a fully fledged CAD kernel. If you need a powerful, open source CAD kernel we recommend you to check out the C++ [OpenCascade](https://www.opencascade.com/doc/occt-7.4.0/overview/html/index.html) library. If you feel that OpenCascade is too complex or you are not sure how to start using it, [this repository](https://github.com/RoberAgro/primer_open_cascade) might be useful for you!

📚 **Documentation**: [https://turbo-sim.github.io/nurbspy/](https://turbo-sim.github.io/nurbspy/)

## Capabilities

`nurbspy` represents both curves and surfaces as rational or non-rational Bézier / B-Spline / NURBS geometries, built directly from control points, weights, and knot vectors. For both, it provides:

- Evaluation of coordinates and arbitrary-order analytical derivatives
- Curvature (and, for surfaces, mean and Gaussian curvature)
- Point projection / point inversion
- Matplotlib-based visualization

Curves additionally offer the tangent, normal, and binormal unit vectors (Frenet-Serret frame), torsion, and arc-length by numerical quadrature. Surfaces additionally offer unit normal vectors, u- and v-isoparametric curves, and constructors for common special surfaces: bilinear, ruled, extruded, revolution, and Coons.

`nurbspy` also works with real and complex data types natively, so shape derivatives can be computed to machine precision with the [complex-step method](https://blogs.mathworks.com/cleve/2013/10/14/complex-step-differentiation/) instead of finite differences. This is useful for shape-optimization problems with many design variables that rely on gradient-based algorithms; `nurbspy` is, to our knowledge, the only Python NURBS package with native complex-number support.

An optional [JAX](https://github.com/google/jax) backend (`nurbspy.jax`) additionally provides automatic differentiation and JIT compilation, see [Installation](#installation).

See the documentation for the [theory](https://turbo-sim.github.io/nurbspy/theory/index.html) behind these capabilities (Bézier, B-Spline, NURBS, and G² continuity) and the [examples](https://turbo-sim.github.io/nurbspy/examples/index.html) gallery, with complete scripts and their numerical and graphical output.


## Installation

`nurbspy` requires **Python 3.11 to 3.13**. Install the latest release from [PyPI](https://pypi.org/project/nurbspy/):

```bash
pip install nurbspy
```

To also install the optional JAX backend and its supporting libraries:

```bash
pip install "nurbspy[jax]"
```

Use `import nurbspy as nrb` for the NumPy interface, or
`import nurbspy.jax as nrb` for the JAX interface. The default installation
does not install JAX. The JAX backend currently uses the CPU.

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

## Contact information

`nurbspy` was originally developed by [Roberto Agromayor](https://www.ntnu.edu/employees/roberto.agromayor) under the supervision of Associate Professor [Lars O. Nord](https://www.ntnu.edu/employees/lars.nord) at the [Norwegian University of Science and Technology (NTNU)](https://www.ntnu.no/) as part of his PhD on turbomachinery shape optimization, and has since been maintained and extended. Please, drop us an email to [roberto.agromayor@ntnu.no](mailto:roberto.agromayor@ntnu.no) if you have questions about the code or you have a bug to report!
