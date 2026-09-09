# Getting started

## Introduction

`nurbspy` represents **Non-Uniform Rational B-Spline (NURBS)** curves and
surfaces. Control points describe the shape, degrees determine the polynomial
bases, knots divide the parameter domain into spans, and weights provide
rational geometry such as exact circular arcs.

The NumPy interface provides curve and surface coordinates, analytical
derivatives, tangents and normals, curvature, curve arc length, point
projection, and Matplotlib plots. Surface constructors include bilinear,
ruled, extruded, revolution, and Coons surfaces. The separate `nurbspy.jax`
subpackage provides differentiable curve and basis-function implementations.
The examples here use `import nurbspy as nrb` unless stated otherwise.

The algorithms follow *The NURBS Book* {cite:p}`NURBS_book`.
This is a geometry library for analysis and parametrization; it does not
provide a complete solid-modeling CAD system.

## Installation for users

Use the following command to install the default NumPy/Numba backend:

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

Continue to the [theory](../theory/index.md) pages for the mathematical
background behind `nurbspy`, or to the [examples](../examples/index.md) to
see it in use.

## Installation for developers

With Git and [Poetry](https://python-poetry.org/docs/#installation) installed,
clone the repository and install the package and development tools:

```bash
git clone https://github.com/turbo-sim/nurbspy.git
cd nurbspy
poetry install --with dev
```

Poetry installs the package in editable mode, so changes to the source are
available without reinstalling. The `dev` dependency group includes testing
and documentation tools. To enable JAX as well:

```bash
poetry install --with dev --extras jax
```

JAX is a package **extra** so it can be selected by both pip and Poetry;
`dev` is a dependency **group** for working on the repository. The full test
suite and the API documentation build both require the JAX extra because they
import `nurbspy.jax`.

### Building the documentation

With Poetry installed, run these commands from the repository root:

```bash
poetry install --with dev --extras jax
poetry run python docs/build_docs.py
```

This generates the API pages, builds the site, opens a browser, and watches
for changes at `http://127.0.0.1:8000`. For a single build:

```bash
poetry run python docs/build_docs.py --no-autobuild
```

The HTML entry point is `docs/_build/html/index.html`. The same commands work
with a virtual environment's Python instead of `poetry run python` when the
development dependencies are already installed.

`docs/build_docs.py` is the shared build entry point. Package settings live in
`docs/conf.py`; all Markdown and reStructuredText documentation lives under
`docs/source/`. Edit narrative pages there. Files in `docs/source/api/` are
regenerated on every build, so edit the Python docstrings to change the API.



### Running the tests

Run the full suite with Poetry from the repository root:

```bash
poetry run pytest
```

The full suite requires the JAX extra because some tests import `nurbspy.jax`. To run a single file, or filter by test name with `-k`:

```bash
poetry run pytest tests/test_nurbs_curve.py
poetry run pytest tests/test_nurbs_curve.py -k derivatives
```

`tests/run_tests.py` runs the three package test files directly through
`pytest.main(["-vv"] + tests_list)` and can be launched without the `poetry
run` prefix once the environment is active:

```bash
python tests/run_tests.py
```
