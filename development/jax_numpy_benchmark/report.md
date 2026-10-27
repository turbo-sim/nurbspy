# NumPy / JAX benchmark

Generated 2026-10-27T18:48:23Z; source commit `4de05a0690b1cc309b2d21649e3afc1081f8e729` plus the documented working-tree basis fix (see recorded hashes).

193 workloads: 182 passed, 11 need review; undefined geometry is recorded separately. Full tables, timing stages and raw samples are in [report.html](report.html) and in the expandable report rows. Local raw data is retained under `outputs/results/`.

## Interpretation

Regular values, derivatives and geometric quantities should be assessed separately from convention differences and fixed-rule integration error. A REVIEW result is not automatically a defective algorithm.

On the 126 passing matched basis/curve/surface workloads, warm JAX is faster in 113 cases versus pure NumPy and 44 versus NumPy + Numba. Small calls can favor Numba. These are local descriptive counts, not a universal speedup; compilation and the number of repeated calls matter.

2D normal convention: NumPy get_normal is the principal Frenet normal; JAX get_normal uses a +90° rotated tangent. Their signs can differ. The separately measured NumPy get_normal_2D / rotated-JAX-tangent adapter checks the equivalent oriented-normal calculation.

Fixed during this study: NumPy/Numba basis derivatives now use an iterative recurrence, return zero above the degree, and retain lower-degree neighbors for non-clamped knots. This removes invalid indexing and the cached recursive calls that produced native process failures. Final timings use corrected source; before-fix evidence is retained. Regression tests cover independent SciPy results, scalars, arrays, complex-step inputs and disk-cache reuse across processes.

Length defaults are different algorithms: NumPy global Gauss-10 under-resolves the smooth and repeated-knot examples; JAX adaptive quadrature and the ODE pass the study threshold. Global JAX fixed Clenshaw-Curtis-41 also under-resolves these examples. Fixed-resolution rules need convergence checks or knot-span splitting; matching a tolerance argument is not equivalent work.

Collapsed-curve frames, curvature and torsion are undefined. The backends return different NaN/zero patterns. These differences require an explicit API policy, not a looser regular-geometry tolerance.

Surface area, whole-geometry energy workflows and NumPy finite-difference sensitivities are benchmark adapters rather than newly added production APIs. Offset surfaces use the package's sampled approximation, not exact geometric offsets.

Prepared tables and dense inverse maps amortize setup only when reused. The map-query row measures reuse separately from full inverse-map construction; table setup includes independent reference evaluation and is not a pure construction microbenchmark.

Projection limitation: NumPy/Numba's single local solve misses a strictly better feasible candidate in the 16-query curve batch; the squared-distance gap is 0.00123385. JAX's sampled initialization passes this batch. Tight optimizer tolerances do not ensure the correct basin; neither finite sampling nor multistart is a universal global certificate.

Inverse-map derivative accuracy: default unit-speed errors are 1.664e-04 (smooth) and 5.567e-04 (repeated knot). Using tol=1e-9 and 64 steps per span reduces them to 5.845e-07 and 1.857e-06. Parameter accuracy alone does not establish unit-speed accuracy. The existing tol=1e-6 is a local solver control, not a global derivative-error guarantee; no source default was changed.

## Selected timings

All times in milliseconds; warm columns are synchronized medians. Blank compilation cells mean host compilation was not isolated. Imports, runtime startup and setup are excluded.

| Workload | NumPy warm | NumPy + Numba warm | JAX warm | JAX lower | JAX compile | JAX first execution |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `curve.value.rational.257` | 0.192 | 0.047 | 0.162 | 58.953 | 86.268 | 1.966 |
| `curve.curvature.rational.257` | 0.544 | 0.190 | 0.300 | 167.199 | 145.536 | 1.637 |
| `surface.value.rational.17` | 0.295 | 0.121 | 0.136 | 31.236 | 99.590 | 0.949 |
| `surface.area.rational.17` | 1.564 | 0.793 | 0.393 | 51.781 | 98.579 | 1.685 |
| `projection_curve.batch.rational.4` | 31.989 | 4.615 | 0.195 | 537.782 | 434.096 | 3.292 |
| `projection_surface.batch.rational.4` | 108.627 | 11.337 | 0.516 | 892.403 | 1,120.247 | 9.193 |
| `workflow.curve.rational.65` | 0.997 | 0.223 | 0.159 | 118.313 | 123.293 | 1.963 |
| `workflow.surface.rational.9` | 6.153 | 2.067 | 0.531 | 121.116 | 260.463 | 6.505 |
| `map_query.arclength.circle.65` | — | — | 0.149 | 286.287 | 141.361 | 0.517 |

## Findings requiring review

- `curve.normal.planar.257`: REVIEW. Independent reference failed: jax
- `length.default.rational.1`: REVIEW. Independent reference failed: numpy, numba
- `length.fixed.rational.1`: REVIEW. Independent reference failed: jax
- `length.default.repeated.1`: REVIEW. Independent reference failed: numpy, numba
- `length.fixed.repeated.1`: REVIEW. Independent reference failed: jax
- `degenerate.normal.collapsed.17`: REVIEW. Independent reference failed: jax
- `degenerate.curvature.collapsed.17`: REVIEW. Independent reference failed: jax
- `degenerate.torsion.collapsed.17`: REVIEW. Independent reference failed: jax
- `projection_curve.batch.rational.16`: REVIEW. Independent reference failed: numpy, numba
- `map_query.unit_speed.rational.65`: REVIEW. Independent reference failed: jax
- `map_query.unit_speed.repeated.65`: REVIEW. Independent reference failed: jax

These deterministic cases are a regression study, not proof of correctness for every input. Read README.md for timing scope, reference tolerances and API coverage. The production basis-derivative fixes are documented in fixes.md and covered by focused regression tests.

## Arc-length absolute error

The native default algorithms differ. Fixed resolution does not imply a tolerance guarantee. NumPy + Numba gives the same length errors as NumPy.

| Curve | Method | NumPy absolute error | JAX absolute error |
| --- | --- | ---: | ---: |
| rational | default | 3.480e-03 | 1.812e-13 |
| rational | adaptive | — | 1.812e-13 |
| rational | ode | — | 3.980e-06 |
| rational | fixed | — | 6.804e-05 |
| circle | default | 2.287e-14 | 2.220e-16 |
| circle | adaptive | — | 2.220e-16 |
| circle | ode | — | 3.627e-07 |
| circle | fixed | — | 0.000e+00 |
| repeated | default | 5.038e-02 | 4.530e-14 |
| repeated | adaptive | — | 4.530e-14 |
| repeated | ode | — | 1.755e-06 |
| repeated | fixed | — | 6.581e-03 |
| line | default | 2.220e-16 | 4.441e-16 |
| line | adaptive | — | 4.441e-16 |
| line | ode | — | 0.000e+00 |
| line | fixed | — | 0.000e+00 |

## Timing figure

See the [embedded timing figure](report.html#warm-execution).

## Validation

Full test suite: 507 passed, 6 skipped, 0 failed, 0 errors. Numerical references and saved-result integrity are checked separately by verify.py. With Numba JIT disabled: 66 additional focused/complex-step checks passed, 6 skipped. JUnit XML is retained locally under `outputs/results/`.

## Dense inverse validation

Setup includes independent references plus map construction; it is not a pure solver timer. Query time includes both parameter evaluation and its unit-speed AD check.

| Curve | Configuration | Maximum parameter error | Maximum unit-speed error | Setup + references (ms) | Warm validation query (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| circle | unit_speed | 6.167e-08 | 2.719e-06 | 3,961.356 | 0.311 |
| rational | unit_speed | 1.948e-06 | 1.664e-04 | 7,872.445 | 0.294 |
| repeated | unit_speed | 7.539e-06 | 5.567e-04 | 7,992.129 | 0.503 |
| rational | unit_speed_tight | 2.626e-09 | 5.845e-07 | 5,836.387 | 0.201 |
| repeated | unit_speed_tight | 2.684e-09 | 1.857e-06 | 5,082.061 | 0.102 |
