# nurbspy

📦 **PyPI package**: [https://pypi.org/project/nurbspy/](https://pypi.org/project/nurbspy/)

📚 **Documentation**: [https://turbo-sim.github.io/nurbspy/](https://turbo-sim.github.io/nurbspy/) *(under construction)* 


## Description
`nurbspy` is a Python package for Non-Uniform Rational Basis Spline (NURBS) curves and surfaces.
The classes and methods were inspired by the algorithms presented in [The NURBS Book](https://doi.org/10.1007/978-3-642-59223-2) and the code was implemented using vectorized [Numpy](https://numpy.org/) functions and [Numba's](http://numba.pydata.org/) just-in-time compilation decorators to achieve C-like speed.

`nurbspy` aims to be a simple NURBS library, not a fully fledged CAD kernel. If you need a powerful, open source CAD kernel we recommend you to check out the C++ [OpenCascade](https://www.opencascade.com/doc/occt-7.4.0/overview/html/index.html) library. If you feel that OpenCascade is too complex or you are not sure how to start using it, [this repository](https://github.com/RoberAgro/primer_open_cascade) might be useful for you!

 
<p style="margin-top:1cm;"> </p>
<p align="center">
        <img src="./docs/images/python_logo.svg" height="120" width="400"/>
<p style="margin-bottom:0cm;"> </p>

<p style="margin-top:0cm;"> </p>
<p align="center">
	<img src="./docs/images/numpy_logo.svg" height="100" width="250"/>
	&emsp; &emsp; &emsp;
	<img src="./docs/images/numba_logo.svg" height="100" width="300"/>
</p>
<p style="margin-bottom:1cm;"> </p>

## Capabilities

`nurbspy` has the following features for NURBS curves:

- Constructors for rational and non-rational Bézier and B-Spline curves
- Methods to evaluate curve coordinates
- Methods to evaluate arbitrary-order derivatives analytically
- Methods to evaluate the tangent, normal, and binormal unitary vectors (Frenet-Serret frame of reference)
- Methods to compute the curvature and torsion
- Methods to compute the arc-length of the curve by numerical quadrature
- Methods to for point projection / point inversion
- Methods to visualize the curve using the Matplotlib library


In addition, `nurbspy` provides the following capabilities for NURBS surfaces:


- Constructors for rational and non-rational Bézier and B-Spline surfaces
- Additional constructors for some common special surfaces:
	- Bilinear surfaces
	- Ruled surfaces
	- Extruded surfaces
	- Revolution surfaces
	- Coons surfaces
- Methods to evaluate surface coordinates
- Methods to evaluate arbitrary-order derivatives analytically
- Methods to evaluate the unitary normal vectors
- Methods to evaluate the mean and Gaussian curvatures
- Methods to compute u- and v-isoparametic curves
- Methods to for point projection / point inversion
- Methods to visualize the surface using the Matplotlib library

In addition,  `nurbspy` can work with real and complex data types natively. This allows to compute accurate (down to [machine epsilon!](https://en.wikipedia.org/wiki/Machine_epsilon)) shape derivatives using the [complex step method](https://blogs.mathworks.com/cleve/2013/10/14/complex-step-differentiation/) and avoid the numerical error incurred by finite-difference derivative approximations. This shape sensitivity information is necessary to solve shape optimization problems with many design variables using gradient based-optimization algorithms. To our knowledge, `nurbspy` is the only Python package that provides the flexibility to work with complex numbers right away.


## Installation

`nurbspy` has the following mandatory runtime dependencies:

 - [numpy](https://numpy.org/) (multidimensional array library)
 - [scipy](https://www.scipy.org/) (scientific computing library)
 - [numba](http://numba.pydata.org/) (just-in-time Python compiler)
 - [pygmo](https://github.com/esa/pagmo2) (optimization library)
 - [matplotlib](https://matplotlib.org/) (visualization library)


`nurbspy` is available on Linux via the [pip](https://pip.pypa.io/en/stable/) package manager. The installation with pip is straightfoward:

```bash
pip install nurbspy
```

You can verify that `nurbspy` was installed successfully with this minimal Python script:

```python
# Nurbspy minimal working example
import nurbspy
nurbspy.minimal_example.run()
```

or by typing this one-liner on your terminal:

```python
python -c "import nurbspy; nurbspy.minimal_example()"
```

## Minimal working examples

### NURBS curves

`nurbspy` can  be used to create Bézier, B-Spline and NURBS curves. The type of curve depends on the arguments used to initialize the curve class. As an example, the following piece of code can be used to generate a degree four Bézier curve in two dimensions:

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

If the installation was succesful, you should be able to see the Bézier curve when you execute the previous code snippet.

<p style="margin-bottom:0.5cm;"> </p>
<p align="center">
        <img src="./docs/images/curve_example.svg" height="400" width="400"/>
</p>
<p style="margin-bottom:0.5cm;"> </p>

Check out the [curve demos](./demos/demos_curves) directory to see more examples showing the capabilities of the library and how to use them.


### NURBS surfaces

Similarly, `nurbspy` can  be used to create Bézier, B-Spline and NURBS surfaces. The type of surface depends on the arguments used to initialize the surface class. As an example, the following code snippet can be used to generate a simple Bézier surface of degree 3 in the u-direction and degree 2 in the v-direction:

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

<p style="margin-top:0.0cm;"> </p>
<p align="center">
        <img src="./docs/images/surface_example.svg" height="500" width="500"/>
<p style="margin-bottom:0.0cm;"> </p>


Check out the [surface demos](./demos/demos_surfaces) directory to see more examples showing the capabilities of the library and how to use them.

## Mathematical background

Check out the [Bézier](./docs/A_Briefing_on_Bezier_Curves_and_Surfaces.pdf), [B-Spline](./docs/A_Briefing_on_B_Spline_Curves_and_Surfaces.pdf), and [NURBS](./docs/A_Briefing_on_NURBS_Curves_and_Surfaces.pdf) notes if you want to learn more about the definition and mathematical properties of these curves and surfaces


## Contact information

`nurbspy` was developed by [Roberto Agromayor](https://www.ntnu.edu/employees/roberto.agromayor) under the supervision of Associate Professor [Lars O. Nord](https://www.ntnu.edu/employees/lars.nord) at the [Norwegian University of Science and Technology (NTNU)](https://www.ntnu.no/) as part of his PhD on turbomachinery shape optimization. Please, drop us an email to [roberto.agromayor@ntnu.no](mailto:roberto.agromayor@ntnu.no) if you have questions about the code or you have a bug to report!
