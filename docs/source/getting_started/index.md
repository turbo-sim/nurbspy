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

Use the following command to install the default NumPy backend:

```bash
pip install nurbspy
```

Numba is optional. To accelerate basis-function evaluation, install:

```bash
pip install "nurbspy[numba]"
```

Numba is used automatically when available. Without it, the same functions
run with NumPy, with unchanged APIs and numerical algorithms but potentially
slower evaluation. If an installed Numba raises an import error, `nurbspy`
warns and falls back to NumPy. The `numba` extra still requires NumPy and
Python versions supported by Numba; omit it to avoid those dependency constraints.

To also install the optional JAX backend and its supporting libraries:

```bash
pip install "nurbspy[jax]"
```

Use `import nurbspy as nrb` for the NumPy interface, or
`import nurbspy.jax as nrb` for the JAX interface. The default installation
does not install Numba or JAX. The JAX extra does not require Numba; use
`pip install "nurbspy[numba,jax]"` to install both extras.
The JAX backend currently uses the CPU.

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

Numba and JAX are package **extras** so they can be selected by both pip and
Poetry; `dev` is a dependency **group** for working on the repository. Use
`poetry install --with dev --extras "numba jax"` to install both extras.
The API documentation build requires the JAX extra because it imports
`nurbspy.jax`.

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
poetry run pytest tests
```

Tests are grouped by what they verify:

| Directory | Purpose |
| --- | --- |
| `tests/geometry/` | Basis, curve, and surface values, derivatives, constructors, and geometric operations. |
| `tests/backends/` | NumPy/Numba complex-step support, Numba compilation, and JAX differentiation, JIT, and batching. |
| `tests/installation/` | Optional-dependency behavior and installed-package smoke checks. |
| `tests/documentation/` | Mathematical derivations and executable documentation examples. |

The `basis_backend` and `curve_backend` fixtures run common numerical checks
against NumPy and JAX. The `surface_backend` fixture uses the existing NumPy
implementation, which Numba accelerates when installed. There is no JAX
surface implementation. Small reusable geometry inputs live in `tests/cases.py`.

NumPy/Numba tests verify complex-step sensitivities with respect to curve and
surface parameters, control points, and weights. JAX tests verify automatic
differentiation instead of requiring complex-step support.

Run a feature group or filter individual tests with `-k`:

```bash
poetry run pytest tests/geometry
poetry run pytest tests/geometry/test_curve_derivatives.py
poetry run pytest tests/backends -k complex_step
```

To exercise both optional backends explicitly:

```bash
poetry install --with dev --extras "numba jax"
poetry run pytest tests --require-backend=numba --require-backend=jax
```

Ordinary runs skip JAX tests if JAX is absent. An installed but broken JAX
backend raises an import error instead of being silently skipped. The
repeatable `--require-backend` option makes a missing or unusable requested
backend fail early, and CI uses it for its optional-extra jobs.

`tests/run_tests.py` discovers the full suite from any working directory and
forwards extra pytest arguments. Once the environment is active, run:

```bash
python tests/run_tests.py
python tests/run_tests.py -k projection
```

### Bumping the version

[bump-my-version](https://github.com/callowayproject/bump-my-version) updates
the version in `pyproject.toml` and `nurbspy/__init__.py` together, then
commits and tags the result, as configured in `.bumpversion.toml`:

```bash
poetry run bump-my-version bump patch   # 1.2.4 -> 1.2.5
poetry run bump-my-version bump minor   # 1.2.4 -> 1.3.0
poetry run bump-my-version bump major   # 1.2.4 -> 2.0.0
```

Preview the changes first with `--dry-run --verbose` (add `--allow-dirty` if
the working tree is not clean); pass `--no-commit --no-tag` to only edit the
files. Push the resulting commit and tag with:

```bash
git push origin main --tags
```
