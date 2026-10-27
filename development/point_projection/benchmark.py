"""Small, reproducible benchmark of the installed production JAX projector.

Run from the repository root:
    python development/point_projection/benchmark.py --smoke

Each case runs in a fresh process to separate lowering, compilation and first
execution. This module contains adapters and measurements, never a replacement
projection algorithm. Correctness regressions belong in tests/backends/.
"""

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

import jax
import jax.numpy as jnp
import numpy as np
import nurbspy.jax as nrb

from cases import curve_cases, numpy_evaluator, surface_cases
from extra_cases import extra_cases, extra_reference


def all_cases():
    return curve_cases() | extra_cases() | surface_cases()


def projector(case):
    """Return a production adapter with geometry passed as dynamic arrays."""
    if case.kind == "curve":
        def project(P, W, U, queries):
            curve = nrb.NurbsCurve(P, W, case.degree_u, U)
            return curve.project_points(queries, tol=1e-10)
        return project

    def project(P, W, U, V, queries):
        surface = nrb.NurbsSurface(P, W, case.degree_u, case.degree_v, U, V)
        return surface.project_points(queries, maxiter=100, gtol=1e-10, ftol=1e-14)
    return project


def distance_check(case, output):
    """Compare distance, allowing nonunique valid parameter representations."""
    _, reference = extra_reference(case)
    parameters = np.asarray(output)
    columns = parameters[:, None] if case.kind == "curve" else parameters.T
    evaluate = numpy_evaluator(case)
    distances = np.array([np.sum((evaluate(uv)[0]-query)**2)
                          for uv, query in zip(columns, case.queries.T)])
    lower = [case.knots_u[case.degree_u]]
    upper = [case.knots_u[-case.degree_u-1]]
    if case.kind == "surface":
        lower.append(case.knots_v[case.degree_v])
        upper.append(case.knots_v[-case.degree_v-1])
    finite = np.all(np.isfinite(columns), axis=1) & np.isfinite(distances)
    in_domain = np.all((columns >= np.asarray(lower)-1e-12) &
                       (columns <= np.asarray(upper)+1e-12), axis=1)
    gaps = distances-reference
    threshold = 1e-8*np.maximum(1., np.abs(reference))
    accurate = finite & in_domain & (np.abs(gaps) <= threshold)
    return dict(parameters=parameters.tolist(), squared_distances=distances.tolist(),
                reference_squared_distances=reference.tolist(),
                objective_gaps=gaps.tolist(), accurate_count=int(np.sum(accurate)),
                finite_count=int(np.sum(finite)), in_domain_count=int(np.sum(in_domain)),
                max_absolute_gap=float(np.max(np.abs(gaps))) if np.all(finite) else None)


def measure(case, repeats):
    values = [case.points, case.weights, case.knots_u]
    if case.kind == "surface":
        values.append(case.knots_v)
    args = tuple(jnp.asarray(value) for value in values+[case.queries])
    jax.block_until_ready(args)
    function = jax.jit(projector(case))
    row = dict(case=case.name, kind=case.kind, query_count=case.queries.shape[1],
               note=case.note, error=None)
    try:
        start = perf_counter()
        lowered = function.lower(*args)
        row["trace_lower_s"] = perf_counter()-start
        start = perf_counter()
        compiled = lowered.compile()
        row["compile_s"] = perf_counter()-start
        start = perf_counter()
        output = jax.block_until_ready(compiled(*args))
        row["first_execution_s"] = perf_counter()-start
        row["cold_kernel_s"] = sum(row[key] for key in
                                   ("trace_lower_s", "compile_s", "first_execution_s"))
        jax.block_until_ready(compiled(*args))
        samples = []
        for index in range(repeats):
            # Changing queries prevent a benchmark of a constant closed-over
            # result. Geometry remains a dynamic executable argument as well.
            changed = args[:-1]+(args[-1]+(index-repeats//2)*1e-7,)
            jax.block_until_ready(changed)
            start = perf_counter()
            jax.block_until_ready(compiled(*changed))
            samples.append(perf_counter()-start)
        row.update(warm_median_s=float(np.median(samples)),
                   warm_min_s=float(min(samples)), warm_max_s=float(max(samples)),
                   warm_samples_s=samples)
        # Independent host-side reference work is deliberately outside timings.
        row.update(distance_check(case, output))
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
    return row


def environment():
    source_dir = Path(nrb.__file__).resolve().parent
    source_files = sorted(source_dir.glob("*projection*.py"))
    source_files += [source_dir/"nurbs_curve.py", source_dir/"nurbs_surface.py"]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()
    return dict(python=platform.python_version(), platform=platform.platform(),
                versions={name: metadata.version(name) for name in
                          ("nurbspy", "jax", "jaxlib", "equinox", "numpy", "scipy")},
                devices=[str(device) for device in jax.devices()],
                x64=bool(jax.config.x64_enabled), persistent_compilation_cache=False,
                git_head=head,
                installed_source=str(source_dir),
                source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in source_files},
                benchmark_sha256={name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                                  for name in ("benchmark.py", "cases.py", "extra_cases.py")})


def write_report(record):
    rows = []
    for row in record["cases"]:
        if row["error"]:
            rows.append(f"| {row['case']} | Error | — | — | — | — |")
            continue
        agreement = f"{row['accurate_count']}/{row['query_count']}"
        rows.append(f"| {row['case']} | {agreement} | {row['trace_lower_s']:.3f} | "
                    f"{row['compile_s']:.3f} | {1e3*row['first_execution_s']:.3f} | "
                    f"{1e3*row['warm_median_s']:.3f} |")
    versions = record["environment"]["versions"]
    report = r"""# Production point projection: benchmark and limits

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
"""+"\n".join(rows)+rf"""

Environment: Python {record['environment']['python']}, JAX {versions['jax']},
Equinox {versions['equinox']}; devices: {', '.join(record['environment']['devices'])}.
Raw per-query distances, warm ranges, environment and production source hashes
are retained in `results/smoke.json`. Hashes identify the measured working tree;
the Git HEAD is recorded separately.

## Accuracy and limitations

- Agreement means the squared-distance gap to the independent reference is at
  most $10^{{-8}}\max(1,|d_{{ref}}^2|)$, with finite parameters in the actual knot
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
"""
    (HERE/"report.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--smoke", action="store_true", help="Two small curve/plane batches")
    selection.add_argument("--all", action="store_true", help="All retained fixtures, including stresses")
    selection.add_argument("--case", action="append", choices=sorted(all_cases()))
    parser.add_argument("--warm-repeats", type=int, default=7)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.warm_repeats < 1:
        parser.error("--warm-repeats must be positive")
    if options.worker:
        print(json.dumps(measure(all_cases()[options.worker], options.warm_repeats)))
        return
    names = options.case or (list(all_cases()) if options.all else
                            ["line_endpoints", "plane_bounds"])
    smoke = not options.case and not options.all
    record = dict(environment=environment(), cases=[])
    for name in names:
        print(f"Production projection: {name}", flush=True)
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                 "--worker", name, "--warm-repeats", str(options.warm_repeats)],
                                cwd=ROOT, text=True, capture_output=True, check=True)
        row = json.loads(result.stdout.strip().splitlines()[-1])
        record["cases"].append(row)
        print(row["error"] or f"  distance agreement {row['accurate_count']}/{row['query_count']}",
              flush=True)
    directory = HERE/"results"
    directory.mkdir(exist_ok=True)
    output = directory/("smoke.json" if smoke else "selected.json")
    output.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    if smoke:
        write_report(record)
    print(f"Saved {output.relative_to(ROOT)}", flush=True)
    if smoke and any(row["error"] or row.get("accurate_count") != row["query_count"]
                     for row in record["cases"]):
        raise SystemExit("Production projection smoke check failed; inspect the saved report/JSON")


if __name__ == "__main__":
    main()
