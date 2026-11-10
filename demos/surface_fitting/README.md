# Surface-fitting demos

Each script performs one fit and opens a Three.js viewer. Blue is the returned
fit; transparent orange is the original rational surface, and dark dots show the
target points used for fitting. All four scripts use a 16 x 16 fitting control
net and 45 x 45 target samples (2,025 points). The original rational patch keeps
its independent 4 x 4 definition. Each script contains the same target
construction, sample cloud and independent flat start, so it can be read and run
on its own. There are no imports from development
scripts and no `sys.path` modifications.

| Script | Point parameters | Weights | Method |
| --- | --- | --- | --- |
| [01_linear.py](01_linear.py) | Known and fixed | Fixed to one | Linear least squares; the rational target need not be reproduced exactly |
| [02_rational_weights.py](02_rational_weights.py) | Known and fixed | Optimized | Pounce interior point with exact sparse JAX Hessian |
| [03_joint_parameters.py](03_joint_parameters.py) | Optimized jointly with the surface | Optimized | Joint exact sparse-Hessian IPM |
| [04_alternating_projection.py](04_alternating_projection.py) | Updated by existing NurbsPy projection | Optimized | Alternating surface IPM and point projection |

Install the package in your environment. Cases 02-04 also need the optional JAX
and Pounce dependencies. From the repository root:

```powershell
python -m pip install -e ".[jax]" "pysolver-view[pounce]>=0.8.2"
python demos/surface_fitting/01_linear.py
python demos/surface_fitting/02_rational_weights.py
python demos/surface_fitting/03_joint_parameters.py
python demos/surface_fitting/04_alternating_projection.py
```

Case 01 reports a direct linear solve. Cases 02-04 print native optimization
convergence; case 04 also reports its outer cycles. A budget exit is not called
convergence. The displayed surface is the actual returned solution in either
case. Each `main(show=False)` returns `(result, figure)` without opening a window,
which is useful for automated runs and optional export.

The reusable API is `nrb.fit_surface`. Point positions have shape `(N, dimension)`.
`parameters=uv` supplies fixed labels. `initial_parameters=uv_guess` supplies
starting guesses for a geometric fit, and may be omitted to initialize using
projection onto the initial surface. Cases 03/04 deliberately omit the target's
generating labels from the fitting call.

The bounds and weak regularization in 03/04 are explicit choices for these
unit-sized patches. The library supports the supplied surface's degrees, control
net and knot domain; it does not assume this target or a graph over x/y.

The [grid-refinement studies](../../development/surface_fitting/README.md) and
their standalone HTML reports remain in `development/surface_fitting/`. The
custom area/orientation-constrained case also remains there.
