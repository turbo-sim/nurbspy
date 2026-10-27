# NumPy / JAX geometry benchmark

This standalone development study compares the production `nurbspy.numpy` and
`nurbspy.jax` backends. It checks numerical results, reports failures without
hiding them, and separates compilation from execution where the API permits it.
It does not replace the correctness suite or add timing assertions to `tests/`.
One clear basis-derivative defect discovered during the study was fixed in
production and regression-tested; see [fixes.md](fixes.md).

Open **[report.html](report.html)** for the interactive, offline report and
[report.md](report.md) for the generated findings. No web server or external
JavaScript library is needed. Expand table rows to see tolerances and samples.

## Reproduce

From the repository root, with the project environment and optional JAX/Numba
dependencies installed:

```bash
poetry run python development/jax_numpy_benchmark/scripts/benchmark.py
```

On Windows the project interpreter can also be invoked directly:

```powershell
.venv\Scripts\python.exe development\jax_numpy_benchmark\scripts\benchmark.py
```

Useful options:

```bash
# Seven repetitions by default; this study's saved run used five.
poetry run python development/jax_numpy_benchmark/scripts/benchmark.py --repeats 5
# Re-render without calculating geometry.
poetry run python development/jax_numpy_benchmark/scripts/benchmark.py --report-only
# Diagnose a subset, retaining other recorded workloads.
poetry run python development/jax_numpy_benchmark/scripts/benchmark.py --select projection
# Omit optional Numba comparison.
poetry run python development/jax_numpy_benchmark/scripts/benchmark.py --profiles numpy jax
```

A full run replaces the selected backend's records. A selective rerun replaces
only matching records and writes its environment/provenance separately. Always
rerun the entire study after source or package changes when making an overall
performance claim. Raw records are written after each workload so a compiler or
native crash does not discard completed results.
The saved study includes a full run plus a selective JAX run for the two
`unit_speed_tight` configurations. Its separate rerun manifest records the
unchanged production source hashes and the diagnostic settings. `outputs/results/report_manifest.json`
records the scripts and input result files used to regenerate the report.

## What is covered

| Family | Calculations and validation |
| --- | --- |
| Basis | Values and derivatives 1–4, partition of unity / zero derivative sum; ordinary and repeated knots, including endpoints |
| Curves | Rational, polynomial, 2D and repeated-knot values, derivatives 1–4, speed, tangent, normal, binormal, curvature, 3D torsion; scalar/small/large sample sets |
| Arc length | Both native defaults, JAX adaptive quadrature, fixed quadrature, ODE and prepared quadrature; straight line, circle, smooth and repeated-knot curves |
| Surfaces | Rational patch, exact plane and paraboloid; paired flattened tensor samples; first, second and higher mixed partials, unit normals, signed mean and Gaussian curvature |
| Surface area | Derived tensor Gauss integration of $\|S_u\times S_v\|$; Gauss-24 versus independent Gauss-64, with a Gauss-48 convergence check; not a native API |
| Projection | Four off-geometry queries per curve/surface; NumPy scalar loops versus JAX batches, finite/in-domain parameters and independent squared-distance checks |
| Construction | Curve merge, circular arc, surface isocurves, attachment in both directions, sampled approximate offset |
| Surface generators | Bilinear, ruled, extruded, revolution and Coons constructions with rational boundary curves |
| Workflows | Sampled curve/surface geometry energies plus integration; whole-control-net sensitivities via JAX AD versus NumPy finite differences |
| JAX-only | Prepared curve/surface basis evaluations, derivatives, curvature, factorized surface grid and arc quadrature; arc-length and monotone-coordinate inverse maps |
| Display | Headless Matplotlib curve/surface rendering and artist counts; not pixel equivalence |
| Degeneracy | Collapsed-curve normal, curvature and torsion; undefined results explicitly classified |

“Slopes” are represented by parameter derivatives $C'(u)$, $S_u$, $S_v$ and
speed. A Cartesian slope such as $dy/dx=C'_y/C'_x$ is singular where $C'_x=0$;
it is not an additional native backend method.

Low-level coordinate and derivative functions are exercised by their public
curve/surface wrappers. Helper variants, every plotting style, all invalid-input
policies, all knot degrees and every optimizer setting are not individually
benchmarked. The study cannot establish equivalence for every possible input.
API inventory and explicit remaining gaps are recorded in `coverage.md`.
Prepared surface projection is also timed as a reusable geometry snapshot.
Control-point and weight sensitivities are measured separately. The additional
oriented 2D normal workload compares NumPy's `get_normal_2D` with an explicitly
rotated JAX tangent, distinguishing that operation from the Frenet normal.
Additional stress cases cover degree-zero and quintic bases/curves, shifted
parameter domains, strongly unequal positive weights, repeated surface knots
and larger projection batches.
Projection-loss sensitivities are checked against an independent envelope
derivative. Dense arc-length inverse maps are checked against independent
cumulative-length inversions and $\|C'(u(s))\|\,du/ds=1$ for circle, smooth and
repeated-knot curves. Non-clamped derivative comparisons use the left limit at
the upper active boundary, where highest derivatives can jump.

## Timing protocol

- Backends run **serially** in separate processes on the CPU, in float64.
  OpenMP/BLAS thread environment variables are set to one. XLA may still use its
  own worker pool; this is not a guarantee of single-threaded CPU execution.
- `numpy` uses `NUMBA_DISABLE_JIT=1` before importing the production backend.
  `numba` uses the same backend with its optional Numba acceleration enabled.
  Numba is not an independent geometry implementation.
- Numba starts with an isolated disk cache. Compiled helper specializations are
  reused within that worker, so first-call timings depend on workload order.
  Unsafe above-degree basis requests are isolated in disposable child processes
  to prevent unchecked native indexing from corrupting the main worker.
- JAX initializes its runtime before workload timing. For transformable APIs,
  `jax.jit(function).lower(*dynamic_inputs)` and `.compile()` are measured
  separately. Persistent compilation caching is disabled; in-process caches are
  cleared between workloads. These are fresh workload compilations in an
  initialized runtime, **not fresh process startup measurements**.
- Geometry arrays are dynamic arguments. Degrees, topology, quadrature nodes
  and prepared basis tables are intentionally static. This avoids compiling an
  entire constant answer. Sample arrays are dynamic for ordinary evaluations.
  Prepared projection and inverse maps intentionally capture a geometry
  snapshot; query points/distances remain dynamic. Rebuild these snapshots
  after changing geometry.
  Geometry-independent Gauss nodes and tensor grids are cached for both
  backends, so NumPy's warm timings do not repeatedly rebuild a rule that JAX
  would hoist out during tracing.
- Every timed JAX result is synchronized with `jax.block_until_ready`. The first
  executable call and every warm sample include Python dispatch and output
  synchronization. Inputs are already device-resident; host/device transfers
  and imports are excluded.
- Host constructors, plotting, native default length calls, inverse-map builds
  and AD/finite-difference comparison adapters have first-call/warm timings.
  Their internal JIT work is included in first call rather than artificially
  split into a separate compiler number. A blank compiler cell means **not
  measured separately**, not zero compilation cost.
- Setup is outside execution timing and includes fixture preparation,
  independent reference evaluation and prepared-table construction. It is not a
  pure table-construction measurement and is disclosed in expanded rows.
- Warm median and individual samples are saved. No timing is a pass/fail gate.
  Small sample counts, CPU frequency changes, thermal effects and other machine
  activity limit precision. Break-even estimates exclude startup and setup.

## Accuracy protocol

Independent references use SciPy `BSpline` with homogeneous control points and
separate rational quotient recurrences. They do not import production basis or
geometry calculations. Integration references use tighter SciPy quadrature or
higher-order tensor Gauss rules. This is numerical reference agreement, not a
formal proof or an interval-certified error bound.

Finite results pass elementwise when

$$|x-x_{\mathrm{ref}}|\leq a_{\mathrm{tol}}+r_{\mathrm{tol}}|x_{\mathrm{ref}}|.$$

Defaults are $a_{\mathrm{tol}}=r_{\mathrm{tol}}=2\times10^{-9}$, length/inverse-map
checks use $3\times10^{-6}$, projection squared-distance checks use
$a_{\mathrm{tol}}=2\times10^{-9}$ and $r_{\mathrm{tol}}=2\times10^{-8}$, and
finite-difference sensitivities use $a_{\mathrm{tol}}=2\times10^{-4}$ and
$r_{\mathrm{tol}}=3\times10^{-5}$. These are study acceptance thresholds, not new
solver guarantees. The report exposes maximum absolute and tolerance-scaled
errors, so results near a threshold remain visible.

Projection references search many curve subintervals, or 25 surface starts plus
edges/corners, with tighter SciPy minimization settings. They are not global
optimality certificates. Parameters are checked for finiteness and bounds;
cross-backend acceptance uses squared distance because minimizers can be
nonunique. Native methods use the same requested convergence controls where
possible, but their stopping conditions and initialization algorithms differ.

NaN is **not** accepted as agreement for regular cases. Collapsed-geometry
frames and curvature/torsion are mathematically undefined; their nonfinite
patterns are recorded separately from valid finite results.

## Files and maintenance

The root keeps the files intended for reading and publishing visible:

```text
jax_numpy_benchmark/
  README.md       # Methodology and reproduction commands
  report.html     # Standalone interactive report, with embedded data/figures
  report.md       # Readable numerical findings and timing tables
  coverage.md     # Public API coverage inventory
  fixes.md        # Production fixes and remaining limitations
  scripts/        # Maintained benchmark, references and report-generation code
  outputs/        # Generated local files; ignored by Git
    results/      # JSON, NPZ, environment/source hashes and JUnit XML
    logs/         # Worker and diagnostic logs
    figures/      # Exported figures; HTML embeds its own figure
    qa/           # Local screenshots and browser validation artifacts
    cache/        # Temporary compiler/bytecode caches
```

Commit the scripts and readable root reports. Everything under `outputs/` is
local and excluded from Git, including intermediate numerical results, logs,
compiler caches and QA screenshots. The HTML embeds its measurements, figure
and environment details, so it remains usable without `outputs/` after cloning
or publishing. The Markdown report links to that embedded figure.

`--report-only` and `scripts/verify.py` require the local result files under
`outputs/results/`. A checkout containing only tracked files can display the
saved reports immediately; regenerating or verifying them requires running the
benchmark to recreate those local files. Worker logs are captured automatically
under `outputs/logs/`. Any manual pytest/JUnit or browser output should also be
written under the corresponding `outputs/` subdirectory.

The scripts share their directory definitions in `scripts/paths.py`:

- `cases.py`: deterministic fixtures and identical constructor inputs.
- `references.py`: independent SciPy evaluations/integration/projection.
- `workloads.py`: registry, production API adapters and derived workflows.
- `benchmark.py`: serial workers, timing, numerical checks and log capture.
- `report.py`: standalone HTML and Markdown findings, using saved results.
- `inventory.py`: source-based public API coverage inventory.
- `verify.py`: analytic reference and saved-result integrity self-checks.

```bash
poetry run python development/jax_numpy_benchmark/scripts/verify.py
```

This study stays out of `tests/`; confirmed production fixes receive small
correctness regressions there. Do not copy the timing harness into the runtime
package.
