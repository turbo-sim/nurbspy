# Production point projection: benchmark and limits

The benchmark imports `nurbspy.jax` and measures the same public methods used by
applications. No alternative projector is maintained in this directory.

## Algorithm

For each query, evaluate a knot-aware grid of **positions**, select the closest
sample, and run one bounded squared-distance minimization from that parameter:

$$u_0=\arg\min_{u_i\in G}\|C(u_i)-Q\|^2,\qquad
u^*=\operatorname{LocalMin}_{u\in[U_p,U_{-p-1}]}\tfrac12\|C(u)-Q\|^2.$$

Surfaces use the corresponding tensor grid and one two-parameter refinement.
Geometry samples are shared across a query batch. The refinement uses analytic
first/second geometry derivatives, a safeguarded descent step and a bounded
line search. Sampled candidates remain available if refinement cannot improve
the objective. This avoids compiling or executing a local solver per sample.
The default budgets are 64 samples per curve span and 8 per surface axis span;
large positive weight ratios also inform logarithmic endpoint clustering.
Constrained stationarity is scaled by physical tangent lengths. `tol` now
reaches the curve solver, while an explicit `gtol` overrides it. Unconverged
iteration exhaustion or stagnation raises instead of silently returning a
failed local iterate.

## Measured smoke check

These CPU float64 measurements are a small implementation check, not a speedup
comparison or a universal performance claim. Each case uses a fresh worker,
disabled persistent compilation caches, and synchronized device results.
Geometry and queries are executable arguments. Warm medians use changing
queries and exclude reference work. First execution is the first call of the
compiled executable; cold kernel cost also includes trace/lower and compile.
Python startup/imports and input placement are excluded.

| Batch | Distance agreement | Trace/lower [s] | Compile [s] | First exec [ms] | Warm median [ms] |
| --- | --- | --- | --- | --- | --- |
| line_endpoints | 16/16 | 0.455 | 0.346 | 3.961 | 0.220 |
| plane_bounds | 6/6 | 0.832 | 0.960 | 9.102 | 0.403 |

Environment: Python 3.12.16, JAX 0.12.0.1,
Equinox 0.13.8; devices: cpu:0.
Raw per-query distances, warm ranges, environment and production source hashes
are retained in `results/smoke.json`. Hashes identify the measured working tree;
the Git HEAD is recorded separately.

## Accuracy and limitations

- Agreement means the squared-distance gap to the independent reference is at
  most $10^{-8}\max(1,|d_{ref}^2|)$, with finite parameters in the actual knot
  domain. Different parameters can correctly represent the same closest point.
- Finite sampling can miss a narrow or competing basin, especially with extreme
  rational weights. A converged local minimum does not certify global distance.
  More samples can help but do not provide a general guarantee.
- Tensor grids grow with both surface axes; sample/query distance work grows
  with the number of samples and queries. Large point clouds may need batching.
- Implicit derivatives require an isolated smooth minimum, a well-conditioned
  reduced distance Hessian, and a stable active set. Ties, collapsed geometry,
  nonsmooth knots and boundary transitions can invalidate ordinary derivatives.
- Independent curve references use SciPy spline evaluation, dense knot-span
  bracketing and scalar stationary roots. Plane and extrusion references are
  analytic/separable. Compressed rational stress cases use polynomial roots
  in an independent regular parameter. These are numerical references rather
  than interval-certified solutions for arbitrary NURBS.

## Reproduce

```powershell
python development/point_projection/benchmark.py --smoke
python development/point_projection/benchmark.py --case folded_nonuniform --case folded_extrusion
python development/point_projection/benchmark.py --all
```

The smoke command refreshes this report and its raw JSON. Selective/full runs
write separate JSON without overwriting the short retained smoke report.
Production correctness and derivative regressions live in
`tests/backends/test_jax_projection_regressions.py`.
