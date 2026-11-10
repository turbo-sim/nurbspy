# Surface fitting

`nurbspy.fit_surface` fits one NURBS patch to sampled positions. The supplied
initial surface defines the control-net size, degrees and knot vectors. The
returned surface is independent of the initial geometry and uses its NumPy or
JAX backend.

The API separates two choices: whether point parameters are known, and whether
rational weights are optimized. It supports known-label linear or rational
fitting, joint surface/parameter fitting, and alternating projection/refitting.
It keeps the patch topology, degrees and knots fixed throughout the solve.

## Dependencies

Unbounded fixed-parameter fitting with fixed weights uses NumPy/SciPy and does
not need JAX or Pounce. Fits that optimize weights, point parameters, or have
finite bounds on free control coordinates use optional JAX derivatives and Pounce
interior point with exact sparse Hessians. For a source checkout:

```powershell
python -m pip install -e ".[jax]" "pysolver-view[pounce]>=0.8.2"
```

This is a Python-driven fitting operation. JAX differentiates its numerical
objective; the complete Pounce solve is not itself a `jax.jit` operation or a
function to differentiate through with `jax.grad`.

## Known point parameters

Let `points` have shape `(N, dimension)` and `uv` have shape `(N, 2)`, with one
parameter pair for each position. The parameters lie in the initial surface's
actual knot domain, which need not be the unit square.

```python
import nurbspy as nrb

# initial_surface is an existing NURBS surface with the desired degrees/knots.
result = nrb.fit_surface(
    points,
    initial_surface,
    parameters=uv,
    optimize_weights=False,
    print_convergence=True,
)
fitted = result.surface
```

With fixed weights and fixed point parameters, rational basis values are known
and fitting the control coordinates is linear least squares. Fixed weights do
not have to be one. A unit-weight fit is a B-spline approximation.

To also optimize positive rational weights:

```python
result = nrb.fit_surface(
    points,
    initial_surface,
    parameters=uv,
    optimize_weights=True,
    weight_bounds=(0.2, 5.0),
    print_convergence=True,
    record_history=True,
)
```

These calls hold every supplied parameter pair fixed. They minimize 3D
correspondence errors and do not project the observations onto the changing
surface. Optimized positive weights use logarithmic variables; the fitter
removes their common scaling ambiguity without changing the initial geometry.
The example weight bounds are optional modelling choices, not data inferred
from the cloud.

`control_mask` and `weight_mask` select free entries (`True`) in the corresponding
control/weight arrays. `control_bounds` and `weight_bounds` are lower/upper pairs
that broadcast to those arrays. If all weights are initially free, the fitter
normalizes them by `W00` and fixes that normalized weight to one; weight bounds
then apply to the normalized values. An already fixed weight preserves the
initial common weight scale. Finite bounds on free control coordinates select
the Pounce route even when weights and point parameters are fixed.

## Unknown point parameters

An unlabeled cloud needs parameter estimates in addition to an initial patch.
Choose `correspondence="joint"` to optimize the surface and all point parameters
in the same nonlinear problem:

```python
result = nrb.fit_surface(
    points,
    initial_surface,
    correspondence="joint",
    optimize_weights=True,
    initial_parameters=uv_guess,
    print_convergence=True,
    record_history=True,
)
```

`initial_parameters` are guesses that may change. They are distinct from
`parameters`, which specify known, fixed correspondences. If initial guesses
are omitted, the fitter initializes them by projecting onto the initial patch.
A suitable initial patch is still needed; automatic surface topology and patch
construction are separate tasks.

Choose `correspondence="alternating"` to hold parameters fixed during each
surface fit, then project onto the updated geometry:

```python
result = nrb.fit_surface(
    points,
    initial_surface,
    correspondence="alternating",
    optimize_weights=True,
    max_cycles=100,
    print_convergence=True,
    record_history=True,
)
```

Projection reuses NurbsPy's existing public JAX bounded-Newton implementation.
It rebuilds sampled surface positions when geometry changes and reuses compiled
code where possible. It does not introduce another projection solver.

The joint method includes mixed surface/parameter curvature in its exact sparse
Hessian. The alternating method solves smaller blocks, but can require many
cycles. Success of individual surface fits does not establish convergence of
the combined alternating problem. Both methods are local; successful
termination does not certify a global minimum or globally nearest points.

Joint parameter optimization requires positive degrees and continuously
differentiable interior joins. Repeated interior knots are supported when they
retain at least C1 continuity; C0 corners and discontinuous joins require a
different stationarity treatment and are rejected by the joint solver. Fixed
parameter fitting and the existing projector retain their own knot support.
Alternating fitting uses the projector's checked one-sided conditions at C0
corners, where a smooth parameter gradient need not vanish.

## Results and visualization

The result provides the fitted `surface`, final `parameters`, `success`,
`message`, `iterations`, `rms`, `objective`, `timings` and `diagnostics`.
`record_history=True` also retains history for inspection or presentation.
Native observations are distinguished from the authoritative returned solution;
their labels preserve what the optimizer actually exposes.

`timings` separates setup/compilation from the warm solve. A budget exit remains
unsuccessful even if the visible geometry already gives a small error.

Use the existing `Figure` API to compare the fit with a reference:

```python
figure = nrb.Figure(
    result.surface,
    resolution=64,
    name="Fitted surface",
    display={"geometry_color": "#147db3", "control_points": False},
)
figure.add(
    original_surface,
    name="Original surface",
    display={"geometry_color": "#ed8b23", "opacity": 0.25,
             "control_points": False, "control_polygon": False},
)
figure.show()
```

The four scripts in `demos/surface_fitting/` are self-contained single-fit
examples. They use the same rational target: 01 fixes unit weights, 02 optimizes
weights at known labels, 03 fits surface and labels jointly, and 04 alternates
projection/refitting. All start from a separate flat surface and print their
convergence information before opening the viewer.

## Development studies and custom constraints

Grid-refinement runs, analytic reference errors, timing comparisons and
standalone HTML studies remain in `development/surface_fitting/`.

Case 05 there defines its own area and orientation constraints. Variable bounds
and fixed-coordinate choices can be reused, but those geometric restrictions
belong to the particular application. A custom constrained problem must include
the full constraint contribution to its Lagrangian Hessian and the appropriate
sparsity pattern; the ordinary fitting Hessian pattern alone is not sufficient
for arbitrary constraints.

## API reference

```{eval-rst}
.. autofunction:: nurbspy.fit_surface

.. autoclass:: nurbspy.SurfaceFitResult
   :members:
```
