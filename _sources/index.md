# Welcome to nurbspy!

`nurbspy` is a Python package for working with Non-Uniform Rational B-Splines (NURBS), a mathematical representation of parametric curves and surfaces widely used in computer-aided design and engineering.

The package provides a compact interface for constructing and manipulating **Bézier, B-Spline, and NURBS curves and surfaces** directly from control points, weights, and knot vectors. The resulting geometries are represented using NumPy arrays and can be readily integrated into scientific computing workflows.

With `nurbspy`, you can evaluate curves and surfaces at arbitrary parameter values, compute analytical derivatives and geometric quantities such as tangents, normals, and curvature, project points onto parametric geometries, and visualize the results with Matplotlib. The emphasis is on exposing the underlying NURBS mathematics through a small and transparent Python API.

`nurbspy` is **not a CAD kernel**. It does not provide trimming, Boolean operations, boundary-representation solid modeling, topology management, or STEP/IGES import and export. Applications requiring those capabilities are better served by a complete geometry kernel such as OpenCascade. The scope of `nurbspy` is intentionally narrower: **representing, evaluating, and differentiating parametric curves and surfaces in Python** while keeping the geometry directly accessible to numerical code.

## Documentation

The sections below cover installation and basic usage, the mathematical theory behind the implementation, documented examples, and the complete API reference.


::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Getting started
:link: getting_started/index
:link-type: doc

Install `nurbspy`, as a user or as a developer.
:::

:::{grid-item-card} Theory
:link: theory/index
:link-type: doc

Definitions, derivatives, and geometric continuity, from Bézier to NURBS.
:::

:::{grid-item-card} Examples
:link: examples/index
:link-type: doc

Run complete scripts and compare their numerical and graphical outputs.
:::

:::{grid-item-card} API reference
:link: api_reference
:link-type: doc

Browse the classes, constructors, and functions available in the package.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

getting_started/index
theory/index
examples/index
api_reference
```
