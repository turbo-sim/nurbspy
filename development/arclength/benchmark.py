"""Serial, fresh-process benchmarks of production nurbspy.jax arc-length APIs.

No integration or inverse solver is implemented here. Independent SciPy/analytic
references live in cases.py. Ordinary correctness tests are under tests/.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METHODS = {"length": ["adaptive", "ode", "fixed_cc41", "prepared16"],
           "workflow": ["adaptive", "leibniz", "ode", "prepared16"],
           "map": ["dense_ode"]}
CASES = {"length": ["quarter_circle", "smooth_3d", "smooth_partial", "repeated_knot",
                    "narrow_spans", "weight_1e12", "weight_1e16", "collapsed", "rational_stress"],
         "workflow": ["smooth_3d"],
         "map": ["quarter_circle", "smooth_3d", "repeated_knot", "narrow_spans",
                 "near_zero_speed", "endpoint_zero_speed", "interior_zero_speed", "stationary_span"]}


def sync(value):
    import jax
    return jax.block_until_ready(value)


def distribution(samples):
    import statistics
    return {"median": statistics.median(samples), "min": min(samples),
            "max": max(samples), "samples": samples}


def executable_timing(function, args):
    start = time.perf_counter()
    lowered = function.lower(*args)
    lower = time.perf_counter()-start
    start = time.perf_counter()
    executable = lowered.compile()
    compile_time = time.perf_counter()-start
    start = time.perf_counter()
    result = sync(executable(*args))
    first = time.perf_counter()-start
    samples = []
    for _ in range(9):
        start = time.perf_counter()
        sync(executable(*args))
        samples.append(time.perf_counter()-start)
    return executable, result, {"trace_lower_s": lower, "compile_s": compile_time,
        "first_execute_s": first, "warm_s": distribution(samples),
        "cold_kernel_s": lower+compile_time+first}


def length_worker(case, method, tol, workflow):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import nurbspy.jax as nrb
    from cases import reference_length

    P, W = jnp.asarray(case.P), jnp.asarray(case.W)
    curve = nrb.NurbsCurve(P, W, degree=case.degree, knots=case.knots)
    reference = reference_length(case)
    sync(curve)
    jax.clear_caches()
    start = time.perf_counter()
    table = nrb.prepare_arclength(curve, case.u1, case.u2, order=16) if method == "prepared16" else None
    sync(table)
    preparation = time.perf_counter()-start if table is not None else 0.

    def evaluate(P, W, info=False):
        c = nrb.NurbsCurve(P, W, degree=case.degree, knots=case.knots)
        if method == "prepared16":
            return table.value(P, W)
        if method == "fixed_cc41":
            return c.get_arclength_quad(case.u1, case.u2)
        if method == "ode":
            return c.get_arclength_ode(case.u1, case.u2, tol, return_info=info)
        return c.get_arclength(case.u1, case.u2, tol,
            adjoint="leibniz" if method == "leibniz" else "direct", return_info=info)

    target = .9*reference["value"]
    def objective(P, W):
        return .5*((evaluate(P, W)-target)/target)**2
    function = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)) if workflow
                       else lambda P, W: evaluate(P, W, info=True))
    executable, result, timing = executable_timing(function, (P, W))
    timing.update(preparation_s=preparation, cold_total_s=preparation+timing["cold_kernel_s"])
    row = {"status": "ok", "reference": reference, "timing": timing}
    if table is not None:
        row["prepared_bytes"] = sum(x.nbytes for x in jax.tree.leaves(table) if hasattr(x, "nbytes"))
    if not workflow:
        value = float(result.value if hasattr(result, "value") else result)
        row.update(value=value, abs_error=abs(value-reference["value"]),
                   accuracy_pass=bool(np.isfinite(value) and abs(value-reference["value"]) <= max(tol, tol*abs(reference["value"])) ))
        if hasattr(result, "_fields"):
            row["diagnostics"] = {key: float(getattr(result, key)) if key in ("value", "error_estimate")
                                  else int(getattr(result, key)) for key in result._fields}
            if int(result.status) != 0:
                row["status"] = "solver_failure"
        return row

    loss, (gP, gW) = result
    rng = np.random.default_rng(23)
    dP, dlogW = rng.normal(size=P.shape), rng.normal(size=W.shape)
    dP /= np.linalg.norm(dP)
    dlogW /= np.linalg.norm(dlogW)
    predicted = float(jnp.sum(gP*dP)+jnp.sum(gW*W*dlogW))
    def reference_loss(points, weights):
        length = reference_length(case, P=points, W=weights)["value"]
        return .5*((length-target)/target)**2
    h = 1e-5
    fd = (reference_loss(case.P+h*dP, case.W*np.exp(h*dlogW))
          -reference_loss(case.P-h*dP, case.W*np.exp(-h*dlogW)))/(2*h)
    designs = [(P*(1+.01*i), W*(1+.02*jnp.cos(jnp.arange(W.size)+i))) for i in range(8)]
    sync(designs)
    samples = []
    for _ in range(7):
        start = time.perf_counter()
        for points, weights in designs:
            sync(executable(points, weights))
        samples.append(time.perf_counter()-start)
    timing["eight_design_s"] = distribution(samples)
    # Validate the timed design family outside the timer. A fast base-case
    # rule alone is insufficient evidence for changing geometry/weights.
    loss_errors, finite = [], True
    for points, weights in designs:
        value, gradients = sync(executable(points, weights))
        loss_errors.append(abs(float(value)-reference_loss(np.asarray(points), np.asarray(weights))))
        finite &= all(np.isfinite(np.asarray(g)).all() for g in gradients)
    row.update(loss=float(loss), gradient_directional=predicted, reference_directional=fd,
               gradient_relative_error=abs(predicted-fd)/max(abs(fd), abs(predicted), 1e-12),
               gradient_finite=bool(finite), design_max_loss_error=max(loss_errors))
    return row


def map_worker(case, tol):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import nurbspy.jax as nrb
    from cases import reference_length, reference_speed, analytic_inverse
    curve = nrb.NurbsCurve(jnp.asarray(case.P), jnp.asarray(case.W), degree=case.degree, knots=case.knots)
    reference = reference_length(case)
    sync(curve)
    jax.clear_caches()
    start = time.perf_counter()
    mapping, total = curve.reparametrize_by_arclength(tol=tol)
    sync((mapping, total))
    construction = time.perf_counter()-start
    targets = jnp.linspace(0, total, 129)
    sync(targets)
    _, result, timing = executable_timing(jax.jit(lambda s: mapping(s)), (targets,))
    timing.update(construction_s=construction, cold_total_s=construction+timing["cold_kernel_s"])
    builds, complete = [], []
    for _ in range(3):
        start = time.perf_counter()
        inverse, length = curve.reparametrize_by_arclength(tol=tol)
        sync((inverse, length))
        builds.append(time.perf_counter()-start)
        sync(inverse(targets))
        complete.append(time.perf_counter()-start)
    timing["cached_construction_s"] = distribution(builds)
    timing["cached_build_query_s"] = distribution(complete)
    u, s = np.asarray(result), np.asarray(targets)
    lo, hi = case.knots[case.degree], case.knots[-case.degree-1]
    refs = [reference_length(case, u1=lo, u2=float(x)) for x in u]
    residual = np.array([r["value"] for r in refs])-s
    centers = np.linspace(.015, .985, 81)*float(total)
    h = 1e-5*float(total)
    uc, up, um = (np.asarray(mapping(jnp.asarray(centers+offset*h))) for offset in (0, 1, -1))
    speed = reference_speed(case, uc)
    keep = speed > max(reference["value"]*1e-5, 1e-14)
    for knot in np.unique(case.knots[(case.knots > lo) & (case.knots < hi)]):
        keep &= ~((um <= knot) & (up >= knot))
    unit = speed*(up-um)/(2*h)
    exact_u = analytic_inverse(case, np.clip(s, 0, reference["value"]))
    row = {"status": "ok", "reference": reference, "total_length": float(total),
        "length_abs_error": abs(float(total)-reference["value"]), "timing": timing,
        "query_count": len(s), "roundtrip_max_abs": float(np.max(abs(residual))),
        "unit_speed_fd_error": float(np.max(abs(unit[keep]-1))) if keep.any() else None,
        "parameter_max_error": float(np.max(abs(u-exact_u))) if exact_u is not None else None,
        "finite": bool(np.isfinite(u).all()), "monotone": bool(np.all(np.diff(u) >= 0)),
        "in_domain": bool(np.all((u >= lo) & (u <= hi))),
        "reference_warnings": [message for r in refs for message in r["messages"]],
        "map_samples": {"s": s.tolist(), "u": u.tolist()}}
    if not row["finite"] or not row["monotone"] or not row["in_domain"]:
        row["status"] = "validation_failure"
    return row


def worker(kind, name, method, tol):
    import jax
    from cases import get_cases, get_mapping_cases
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_enable_compilation_cache", False)
    row = {"kind": kind, "case": name, "method": method, "comparison_tol": tol,
           "tolerance": tol if method not in ("fixed_cc41", "prepared16") else None}
    try:
        case = next(c for c in get_cases()+get_mapping_cases() if c.name == name)
        detail = map_worker(case, tol) if kind == "map" else length_worker(case, method, tol, kind == "workflow")
        return {**row, **detail}
    except Exception as error:
        expected = (kind == "map" and name in ("endpoint_zero_speed", "interior_zero_speed", "stationary_span", "collapsed")
                    and any(word in str(error) for word in ("positive speed", "zero-length", "stationary/flat")))
        return {**row, "status": "expected_rejection" if expected else "error",
                "reason": f"{type(error).__name__}: {error}"[:4000]}


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def metadata():
    names = ["jax", "jaxlib", "equinox", "diffrax", "optimistix", "quadax", "numpy", "scipy"]
    sources = list((ROOT/"src/nurbspy/jax").glob("*.py"))+list(HERE.glob("*.py"))
    return {"time": time.strftime("%Y-%m-%d %H:%M:%S%z"), "python": sys.version,
        "platform": platform.platform(), "processor": platform.processor(),
        "versions": {name: importlib.metadata.version(name) for name in names},
        "source_parent": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "protocol": {"device":"cpu", "float64": True, "native_thread_limits":1,
                     "persistent_cache":False, "fresh_process_per_cell":True,
                     "ode_length_max_steps":4096, "map_max_steps_per_span":512,
                     "map_minimum_steps_per_span":16}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker", nargs=2)
    parser.add_argument("--kind", choices=METHODS, default="length")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, default=HERE/"results")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(clean(worker(args.kind, *args.worker, args.tol)), indent=2, allow_nan=False))
        return
    if not args.run:
        parser.error("Pass --run to time serial fresh-process workers")
    methods, names = args.methods or METHODS[args.kind], args.cases or CASES[args.kind]
    if any(m not in METHODS[args.kind] for m in methods):
        parser.error("Unsupported method for this workload")
    args.output.mkdir(parents=True, exist_ok=True)
    identity = metadata()
    snapshot = f"environment_{args.kind}_{time.time_ns()}.json"
    (args.output/snapshot).write_text(json.dumps(identity, indent=2), encoding="utf-8")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               JAX_ENABLE_X64="true", JAX_PLATFORM_NAME="cpu", JAX_ENABLE_COMPILATION_CACHE="false", MPLBACKEND="Agg")
    for name in names:
        for method in methods:
            path = args.output/f"{args.kind}_{name}_{method}_{args.tol:g}.json"
            if path.exists() and not args.overwrite:
                print(f"skip {path.name}", flush=True)
                continue
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", name, method,
                       "--kind", args.kind, "--tol", str(args.tol)]
            start = time.perf_counter()
            try:
                process = subprocess.run(command, cwd=HERE, env=env, capture_output=True,
                    text=True, encoding="utf-8", errors="replace", timeout=args.timeout)
                try:
                    row = json.loads(process.stdout)
                except json.JSONDecodeError:
                    row = {"kind":args.kind,"case":name,"method":method,"status":"worker_failure",
                           "stdout":process.stdout}
                if process.stderr:
                    row["stderr"] = process.stderr
                row["returncode"] = process.returncode
            except subprocess.TimeoutExpired:
                row = {"kind":args.kind,"case":name,"method":method,"status":"timeout"}
            row.update(environment_snapshot=snapshot, worker_wall_s=time.perf_counter()-start,
                       comparison_tol=args.tol)
            path.write_text(json.dumps(clean(row), indent=2, allow_nan=False), encoding="utf-8")
            print(f"{path.name}: {row['status']} ({row['worker_wall_s']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
