# Production arc-length benchmark

This directory calls public `nurbspy.jax` APIs. It contains no alternative
integration or inverse solver. Independent analytic and SciPy references live
in `cases.py`; ordinary correctness regressions live in `tests/`.

Read [report.md](report.md) for the retained results and figures, and the
[usage guide](../../docs/source/getting_started/jax_arclength.md) for API details.

```powershell
python development/arclength/benchmark.py --run --kind length
python development/arclength/benchmark.py --run --kind length --tol 1e-9
python development/arclength/benchmark.py --run --kind workflow --tol 1e-9
python development/arclength/benchmark.py --run --kind map
python development/arclength/make_report.py
```

Use an environment satisfying the repository's JAX extra. Select subsets with
`--cases` and `--methods`. Use a new `--output` for a new study; existing cells
are skipped unless `--overwrite` is supplied. Run timed workers serially.
Raw JSON includes failures, accuracy, all samples and immutable environment/
source-hash snapshots. Fixed rules have no requested tolerance even though the
filename carries the comparison's accuracy threshold.

Workers use CPU float64, native-library thread limits of one and disabled
persistent compilation caches. Outputs are synchronized. Interpreter startup,
imports, curve construction and independent references are outside kernel
timings. Length/workflow stages separate preparation, trace/lower, compilation,
first compiled execution and warm execution. Cold totals include these stages.
Nine warm samples and seven eight-design workflow samples are retained.

Inverse construction includes internal compilation, quadrature, ODE span solves
and validation. Query compilation is measured separately for 129 distances.
Cached rebuilds reuse the same geometry and loaded code caches. Queries use
dense interpolation only. Independent partial lengths check round trips;
finite differences of actual returned maps check unit speed. Stencils crossing
knots and low-speed probes are excluded. Sampling is not a global proof.

One fresh worker per cell does not characterize all run-to-run clock/compiler
variation. Compare achieved accuracy alongside observed timing ranges. There
are no timing thresholds in the ordinary test suite. Keep this single canonical
benchmark and dated evidence for maintenance; Git history preserves previous
code. Development artifacts are not installed runtime modules.
