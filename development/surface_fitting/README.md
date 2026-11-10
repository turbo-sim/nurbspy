# Surface-fitting studies

Five related formulations fit the same analytic six-hill/valley terrain. Every
case uses 4x4, 8x8, 16x16 and 32x32 cubic control grids, 45x45 observations,
and an independent flat initial surface with unit weights. Noise is zero by
default. These are development studies of the public `nrb.fit_surface` API;
target construction, refinement loops and reports remain here.

For short, self-contained single-fit scripts, see
[demos/surface_fitting](../../demos/surface_fitting/README.md). Those four demos
compare the same original rational patch with each fitted solution in a viewer.

The [formulation note](FORMULATION.md) gives the equations, unknowns, bounds,
regularization, derivative structure, convergence checks and error definitions.

## Run all five

Install NurbsPy and the optional nonlinear solver dependencies in one environment:

~~~powershell
python -m pip install -e ".[jax]" "pysolver-view[pounce]>=0.8.2"
python development/surface_fitting/run_all.py
~~~

The runner executes one process per demo, sequentially. It writes
[output/index.html](output/index.html), which links all five standalone reports.
Each report embeds its data and Three.js renderer and works without a Python
server, CDN or internet connection.

The nonlinear fitting solvers all use Pounce interior point and exact sparse Hessians.
Point projection reuses NurbsPy's existing batched JAX bounded-Newton API.
No SLSQP, SciPy nonlinear optimizer or limited-memory curvature is used.
NumPy/SciPy still supply ordinary linear algebra, B-spline basis evaluation and
sparse matrix storage. JAX supplies exact double-precision derivatives. Importing
the helpers does not change global JAX precision.

Each launcher sets unset BLAS/OpenMP/Rayon thread budgets to four before numerical
imports. Environment variables set before launching take precedence. Package
versions, thread settings and source hashes are saved with every experiment.
The fingerprints include the installed `nurbspy.fitting` source as well as the
development scripts, so editable-checkout runs retain their actual engine version.

## The five examples

| Case | Script | Correspondences | Method |
| --- | --- | --- | --- |
| 01 | [demo_01_linear_least_squares.py](demo_01_linear_least_squares.py) | Known, fixed u/v | Direct tensor-product linear least squares; weights fixed to one |
| 02 | [demo_02_nonlinear_weights.py](demo_02_nonlinear_weights.py) | Known, fixed u/v | All control coordinates and positive weights; exact sparse-Hessian IPM |
| 03 | [demo_03_joint_parameters.py](demo_03_joint_parameters.py) | Unknown, optimized jointly | All coordinates, weights and per-point u/v; joint exact sparse-Hessian IPM |
| 04 | [demo_04_alternating_projection.py](demo_04_alternating_projection.py) | Unknown, updated by projection | Alternate exact sparse-Hessian surface IPM and existing NurbsPy point projection |
| 05 | [demo_05_constrained_ipm.py](demo_05_constrained_ipm.py) | Known, fixed u/v | Heights/interior weights with area and orientation constraints; exact sparse Lagrangian Hessian |

Individual launchers expose their main settings as constants near the top and
as keyword arguments to main(). They also accept CLI overrides, for example:

~~~powershell
python development/surface_fitting/demo_03_joint_parameters.py --grids 4 8 --show
python development/surface_fitting/demo_04_alternating_projection.py --max-cycles 100
~~~

The supported control grids are 4, 8, 16 and 32 for every case. Each nonlinear
IPM solve has native tolerance 1e-8 unless a tighter inner tolerance is needed
for the joint/projection stationarity checks. The default iteration budget is
1000 per inner solve. Separate PySolverView KKT diagnostics and KKT stopping are
disabled; native solver checks and the explicit formulation-specific audits
remain active. A budget exit is never labeled convergence.

### 02: known u/v

The generating labels are retained from the analytic data generator. They remain
fixed throughout optimization. This case does not project points or update
correspondences, and does not directly fit an arbitrary unlabeled cloud.

Weights are logarithmic, bounded to [0.1, 10], with W00=1. All control coordinates
are free and unbounded. The objective is 1000 times half the mean squared 3D
correspondence error. Reverse AD differentiates the scalar loss; the full sparse
Hessian includes residual-dependent second derivatives. Prepared local basis
tables are reused, with no dense Hessian or full residual Jacobian in the solve.

### 03: joint surface and point parameters

The objective uses point positions, not their true labels. Each point adds two
parameter unknowns over the global domain [0,1]^2. Initial estimates are measured
x/y clipped to the domain, appropriate for this graph-like surface.

All variables are optimized jointly in each IPM pass. Local parameter windows
cover adjacent knot spans so that every possible nonzero within a pass is in a
fixed sparse Hessian pattern. Windows are recentered if their artificial bounds
prevent stationarity on the global domain. This does not alternate projection
and fitting, and does not silently drop mixed Hessian terms.

Planar controls have bounds [-0.25, 1.25], heights [-0.6, 0.6], and weights
[0.1, 10], with W00=1. Weak planar-frame and log-weight regularization stabilize
redundant parameterizations. The default window budget is eight passes. Native
success in a local window alone is insufficient for reported convergence.

### 04: alternating projection and fitting

The objective, bounds and regularization are the same as case 03. A surface
step holds u/v fixed and uses Pounce interior point with an exact sparse Hessian.
A projection step holds the surface fixed and calls the existing public
prepare_surface_projection/project_points API. Its bounded Newton method uses
exact 2x2 Hessians per point and permits support changes over the full domain.
Surface sampling is rebuilt whenever geometry changes; compiled code is reused.
The adapter batches queries to limit memory and adds the preceding fitted
correspondences as valid surface-sample candidates for library initialization.

Each phase is recorded separately, including projection endpoints where the
surface geometry stays unchanged. Both blocks must be stationary at the same
final state. The default budget is 100 cycles; an unfinished alternating method
is reported honestly even if each inner IPM solve converged.

### 05: constrained fitting

All planar control coordinates stay at their initial Greville values and all
boundary weights stay one, preserving the XY outline. Boundary heights are free;
no exact 3D terrain boundary is prescribed. Heights are bounded to [-0.6, 0.6]
and interior weights to [0.5, 2]. The area cap is

$$
A(\mathbf S)\leq1+0.85(A_{\mathrm{terrain}}-1).
$$

This excludes an exact target fit while preserving a feasible flat start. Sampled
planar Jacobians must be at least 0.05. Native Pounce receives a sparse constraint
Jacobian and the full multiplier-weighted Lagrangian Hessian. Area quadrature
and the orientation grid are refined independently for final audits. Sampling
is not a proof against folding between guard points.

## Reports and numerical archives

The Geometry explorer includes control-grid selection, visibility toggles,
initial/final comparison and playback of stored nonlinear states. Camera and
layer choices survive case changes. Reports sample at most 61 stored states per
grid while preserving actual iteration, cycle and phase numbers. No interpolated
or fabricated iterations are inserted. Pounce observation coordinates come from
the last objective evaluation and have unknown acceptance; this is stated in
the report. Authoritative returned solutions are stored separately.

The Grid refinement page contains error charts, observed orders, variable counts,
solver status, compilation/setup/warm timings and a numerical-kernel breakdown.
Cases 01/02/05 use fixed-label errors on an independent 193x193 reference grid.
Cases 03/04 instead use the existing NurbsPy local projector on a 65x65 reference
grid; their fixed-label error is also retained. Initializer candidates include
the library's surface samples and the learned parameters of nearby observed
points, preserving the fitted parameterization. The B-spline comparison uses
the corresponding reference metric. Neither local projection nor native
convergence certifies a global minimum.

Every output folder contains:

- report.html: self-contained interactive report.
- experiment.json: mathematical settings, environment and source hashes.
- metrics.json: errors, timings, counts, stopping reasons and audit results.
- fit_metrics_NN.json: fit checkpoint saved before reference projection.
- observations.npz: the exact shared points and generating labels.
- initial_NN.yaml and fit_NN.yaml: starting and authoritative fitted geometry.
- iterations_NN.npz: stored control nets, weights, RMS, native indices and roles.
- native_history_NN.json: nonlinear native metrics, with phases/passes where relevant.
- correspondences_NN.npz: final point parameters for 03/04.
- timings.txt: readable timing and convergence context for nonlinear cases.

Setup includes optional imports, basis/sparsity construction, compilation and
kernel warmup. Warm solve excludes setup, geometry conversion, final audits and
HTML generation. Case 03 sums joint IPM passes; case 04 sums the surface-fitting
IPM solves and library point projection, recording each part separately. Native
iteration/evaluation counts describe fitting IPM only; the projection API does
not expose its Newton iteration counts. Reference projections are postprocessing
and reported separately. Measurements are one sequential run per grid, not
repeated averages. Reusing compiled problems reduces setup rather than an
already warm solve.

_report.rebuild_report(directory) rebuilds a saved report without repeating fits.
Output paths are anchored at development/surface_fitting/output, independently of
the working directory. The runner does not delete results automatically.

For further warm-time improvements, use the measured kernel breakdown first.
When Hessian assembly dominates, larger HVP batches or direct local-block
assembly are candidates to benchmark. In case 04, carrying native primal/dual
warm-start information between surface solves could reduce repeated barrier
work. When the solver/interface remainder dominates, profile sparse ordering
and factorization before attributing all remaining time to linear algebra.
These are opportunities, not implemented speedups; compilation caching affects
setup rather than an already warm solve.

## Source layout

- _experiment.py: analytic data, reference metrics and a public linear-fit adapter.
- _nonlinear.py: terrain bounds/penalties, report schema and compatibility constructors.
- _joint.py: public joint-fit adapter and a compatibility constructor for regressions.
- _projection.py: compatibility import of the shared existing-projector adapter.
- _alternating.py: public alternating-fit adapter with explicit terrain settings.
- _constrained.py: native sparse constrained Pounce IPM and Lagrangian Hessian.
- _advanced.py: custom case-05 geometry derivatives and area/orientation construction.
- _study.py: archives, matched error metrics and report orchestration.
- _report.py: offline visualization and timing presentation.

The reusable algorithms are in [src/nurbspy/fitting](../../src/nurbspy/fitting/).
The small compatibility constructors keep the development derivative regressions
useful without duplicating a fitting or projection implementation. Case 05 keeps
its application-specific constraints and weighted Lagrangian Hessian here.

The note documents the current formulations and their limitations. Existing
saved outputs remain measurements of their recorded run; moving the algorithms
into `src` does not rewrite those timing results.
