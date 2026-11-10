# Surface fitting to point clouds: assessment and proposed API

Prepared 7 November 2026 for [GitHub issue #2](https://github.com/turbo-sim/nurbspy/issues/2).
Status: historical proposal, superseded by the surface-fitting implementation
in version 1.7.0. The recommendations below record the initial design discussion;
they do not describe the final API. See [Surface fitting](source/surface_fitting.md)
for the implemented `fit_surface` function, `SurfaceFitResult`, supported
backends, and optional nonlinear solver dependencies. No GitHub comments were
posted as part of this assessment.

## Recommendation

Add one public `fit_surface()` function that returns an ordinary NumPy
`NurbsSurface`. Reuse an existing surface as the initial guess and definition of
its degree, knots, control-net shape, and rational weights. Return an optional
diagnostics dictionary rather than introducing a fitting class or result class.

Support two explicitly documented operations through that function:

1. Supplied surface parameters: solve a linear least-squares problem for the
   control-point coordinates, keeping parameters and rational weights fixed.
2. Unknown surface parameters: alternate point projection and control-point
   fitting, starting from the caller's initial surface.

Develop and verify the first operation before the second. A known-parameter
solver alone is useful, but it does not fully address fitting an arbitrary
unordered cloud; that distinction must remain clear in release notes and any
reply to the issue.

## What the issue asks for

The single open issue asks about fitting NURBS surfaces to point clouds,
specifically mentioning the absence of least-squares fitting. It was opened on
22 March 2022; a second user expressed the same interest on 18 April 2025.
Neither specifies whether the data have known parameters, grid connectivity,
boundaries, or a suitable initial surface. Those details materially change the
problem. [Issue and discussion](https://github.com/turbo-sim/nurbspy/issues/2)

## What can be reused

| Existing code | Contribution | Limitation |
| --- | --- | --- |
| `src/nurbspy/numpy/nurbs_basis_functions.py` | Basis values and derivatives | Needs fitting-matrix assembly around it |
| `src/nurbspy/numpy/nurbs_surface.py` | Rational tensor-product evaluation and analytical derivatives | No public surface-fitting function |
| NumPy `project_point_to_surface()` | Closest-point search for one query | Current bounds are hard-coded to `[0, 1]`; initialization is repeated per query |
| JAX `NurbsSurface.project_points()` | Batched projection over actual knot domains | JAX is optional; its local-solver and differentiation assumptions apply |
| `demos/optimization_problems/demo_curve_fitting_point_inversion.py` | Example of fitting with projected correspondences and optimized curve weights | Curve-specific demonstration, not a reusable surface-fitting API |
| `demos/optimization_problems/demo_minimum_bending_energy.py` | Existing example of a basis-matrix least-squares solve | An interpolation baseline for a curve, not surface reconstruction |
| `_control_points.py`, `Point`, and geometry YAML I/O | Consistent coordinate ordering and normal geometry outputs | Query points require their own validation and are not control points |

This is a new numerical operation around existing mathematical primitives, not
a new geometry representation. NumPy and SciPy are sufficient for the proposed
initial implementation. PCL, VTK, Open3D, and a mandatory JAX dependency are not
needed.

## Proposed API

The following signature is illustrative and does not exist yet:

```python
def fit_surface(
    points,
    *,
    initial_surface,
    parameters=None,
    sample_weights=None,
    smoothing=0.0,
    maxiter=50,
    tol=1e-8,
    return_info=False,
):
    ...
```

Proposed public imports are `nurbspy.fit_surface` and
`nurbspy.numpy.fit_surface`, both pointing to one implementation. Do not add
parallel constructor names, class methods, or JAX implementations in this batch.

### Inputs and ownership

- `points`: finite real coordinates with shape `(ndim, N)`, consistent with the
  rest of nurbspy, or a sequence of `nrb.Point` objects. Do not guess whether a
  numerical array needs transposing. Initially target spatial data (`ndim=3`).
- `initial_surface`: a NumPy `NurbsSurface`. Its degrees, knots, rational weights,
  and control-net shape define the approximation space. It supplies an actual
  initial geometric guess when parameters are unknown. It is never mutated.
- `parameters`: finite real coordinates with shape `(2, N)`, ordered as u and v.
  They must lie within the surface's active knot domains, not necessarily
  `[0, 1]`. If provided, they remain fixed. If omitted, the fitter estimates them
  by point projection and updates them during fitting.
- `sample_weights`: optional nonnegative weights with shape `(N,)`. Require a
  positive total weight. Zero-weight points are excluded from fitting and from
  rank checks. These measurement weights are distinct from rational control-point
  weights, which remain fixed and strictly positive.
- `smoothing`: finite nonnegative strength of control-net smoothing. Zero means
  no regularization. This is a control-net penalty, not a physical curvature or
  bending-energy model. Its normalization must be documented.
- `maxiter`, `tol`: outer-iteration controls for unknown parameters; they do not
  alter the direct least-squares operation with supplied parameters. `tol` is a
  relative objective-change threshold, not a guaranteed spatial accuracy.
- `return_info`: false returns the fitted surface; true returns
  `(surface, info)`. Numeric options must reject booleans where inappropriate.

Requiring an initial surface keeps topology and basis choices explicit and reuses
the existing constructor. Automatic initialization can be added later without
changing the output geometry type.

### Example with known parameters

```python
# Proposed API; Q has shape (3, N), uv has shape (2, N).
surface, info = nrb.fit_surface(
    Q,
    initial_surface=template,
    parameters=uv,
    smoothing=0.0,
    return_info=True,
)
surface.to_yaml("fitted_surface.yaml")
```

Only the template's basis and weights matter in this direct solve. Its original
control-point coordinates are not a prior or a constraint.

### Example with an unordered cloud

```python
# Proposed second milestone; initial_surface must already be reasonably close.
surface, info = nrb.fit_surface(
    Q,
    initial_surface=guess,
    smoothing=1e-4,
    maxiter=50,
    return_info=True,
)
```

The result should work with existing evaluation, plotting, serialization, and
export methods because it is a regular `NurbsSurface`.

## Numerical approach

For fixed knots, rational weights, and parameters, define the rational basis

$$
A_{k,ij} =
\frac{N_{i,p}(u_k)M_{j,q}(v_k)w_{ij}}
{\sum_{a,b}N_{a,p}(u_k)M_{b,q}(v_k)w_{ab}}.
$$

Flatten the control net into a matrix $C$ with one row per control point and
three columns. Let $Q$ have one measured point per row. Then predicted points
are $AC$. A proposed normalized objective is

$$
\frac{1}{\sum_k d_k}\sum_k d_k\|(AC)_k-Q_k\|^2
+\frac{\lambda}{m_D}\|DC\|_F^2,
$$

where $d_k$ are measurement weights, $D$ stacks second differences of the control
net in its two directions, and $m_D$ is its number of rows. If no second-difference
rows exist, reject positive smoothing rather than divide by zero. This scaling
makes uniform duplication of observations leave the relative regularization
unchanged; it does not make smoothing independent of control-net refinement.

Solve the augmented least-squares system with a QR/SVD-based routine, for all
coordinate columns together. Avoid forming normal equations. SciPy's
`linalg.lstsq` supplies effective rank and singular values, which are useful for
diagnostics. Recompute residuals explicitly rather than relying on a solver's
conditionally populated residual field. [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html)

For unknown parameters, alternate projection onto the current surface and this
linear update. Evaluate the full regularized objective before accepting an
iteration; reject worsening updates or failed projections. Include an objective
history and distinguish stagnation from successful stationarity. The procedure
finds a local fit and cannot certify globally closest correspondences or a
globally optimal surface.

The NumPy projection path needs review first: use actual active knot domains,
reuse sampling across queries, handle patch boundaries and zero-distance
queries, and report failures. Merely calling its current scalar method in a
loop would inherit its domain limitation and repeated initialization cost.

## Initialization and scope

An unordered point cloud does not specify its u/v correspondence or patch
boundaries. Automatic initialization should therefore be a separate, explicit
future option. For example, PCL's fitting tutorial initializes from PCA under a
rough-planarity assumption, and treats refinement and trimming as additional
operations. That is useful precedent, not a general solution for all shapes.
[PCL fitting tutorial](https://pointclouds.org/documentation/tutorials/bspline_fitting.html)

The proposed first scope is a single open tensor-product patch with fixed
topology and weights. It does not include automatic segmentation, trimming,
hole reconstruction, periodic seams, multi-patch continuity, adaptive knots,
weight optimization, or general watertight reconstruction. A complex closed
object should not silently be forced into this one-patch workflow.

Structured measurement grids are an easier additional input mode because they
provide neighborhood information. A future explicit helper could derive u/v by
averaged chord lengths, with handling for degenerate rows and columns. The first
API should accept already flattened observations and explicit parameters rather
than introduce hidden grid inference.

## Diagnostics and failure behavior

Proposed `info` fields:

| Field | Meaning |
| --- | --- |
| `parameters` | Supplied or fitted `(2, N)` parameter coordinates |
| `residuals` | Euclidean residual for each observation at its reported parameters |
| `rms_error`, `max_error` | Unweighted spatial errors over positive-weight observations |
| `objective`, `objective_history` | Regularized objective, separate from spatial error |
| `rank`, `augmented_rank`, `condition_number` | Data-system rank and augmented-system diagnostics |
| `iterations`, `converged`, `message` | Outcome of the direct solve or iterative procedure |

For fixed parameters, residuals are correspondence errors, not necessarily
shortest distances to the surface. For unknown parameters, they are distances
to the locally projected locations, not certified global minima. Zero-weight
observations may have residuals reported but do not affect error summaries.

Reject nonfinite data, inconsistent shapes, invalid parameter domains, invalid
weights, and underdetermined augmented systems with actionable errors. A large
point count alone does not guarantee sufficient parameter coverage or rank.
Regularization must not conceal insufficient data; report both data and
augmented ranks. For finite but ill-conditioned full-rank systems, report the
condition estimate and issue a warning using a documented threshold.

Failed projection or exhausted outer iterations must not be reported as
convergence. Recommended policy: return the last accepted surface with a
`RuntimeWarning` and `converged=False` when it remains meaningful; raise when no
valid fit exists. Requesting `return_info=True` must not suppress that warning.

## Small feasibility check

A one-off numerical experiment used existing basis functions, an existing
degree-(2,2) rational surface with 4 by 5 control points and positive nonuniform
weights, and NumPy's least-squares solver. No fitting function was added.

| Check | Observed result |
| --- | --- |
| 420 noiseless samples with known parameters | Full rank 20; maximum coordinate error $1.776\times10^{-15}$ |
| Coordinate noise with standard deviation 0.01 | Spatial RMS against the clean surface at 1,000 held-out parameters: 0.003943 |
| All parameter pairs collapsed to one sample location | Rank 1 for 20 unknown control points |

The random seed was 2407. These checks support the fixed-weight linear kernel
and the need for rank diagnostics. They are not evidence of robust unordered-cloud
fitting, generalization across datasets, or production performance.

## Implementation and acceptance plan

1. Add one `numpy/fitting.py` module and root/NumPy exports. Keep new algorithmic
   logic out of the already large `NurbsSurface` class.
2. Implement the known-parameter solve, validation, smoothing, and diagnostics.
   Test exact recovery of polynomial and rational patches, non-unit parameter
   domains, layout/flattening consistency, measurement weights, rank deficiency,
   noise on held-out data, and unchanged inputs.
3. Review and extend NumPy projection for repeated surface queries. Test interior,
   edge/corner, exact-hit, non-unit-domain, degenerate, and failed-solve cases.
4. Implement alternating fitting. Test a perturbed initial surface, accepted
   objective decrease, parameter bounds, failure reporting, deterministic results,
   and sensitivity to poor initialization.
5. Add separate known-parameter and unordered-cloud demos using ordinary imports,
   plus documentation of assumptions, residual meanings, and unsupported cases.
   Use existing plotting; visualization features are not prerequisites.
6. Benchmark representative clouds and control-net sizes before deciding whether
   sparse assembly/solving is necessary. Dense storage is proportional to the
   sample count times the number of control points; large scans need a considered
   sparse path rather than an unlimited-memory promise.

Milestones 1 and 2 deliver parameterized least-squares approximation. Milestones
3 and 4 deliver the initialized unordered-cloud fitting relevant to the original
issue. A public claim of arbitrary point-cloud reconstruction would still be too
broad. Implementation should begin only after the API and scope are agreed.
