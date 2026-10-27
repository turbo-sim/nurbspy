"""Generate the report/figures from retained production API measurements."""

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABELS = {"adaptive": "Adaptive quadrature", "leibniz": "Adaptive + Leibniz AD",
          "ode": "Knot-aware ODE", "fixed_cc41": "Fixed global CC41",
          "prepared16": "Prepared span GL16", "dense_ode": "Dense span ODE"}


def fmt(x, scale=1):
    return "—" if x is None or not math.isfinite(float(x)) else f"{float(x)*scale:.3g}"


def timing(row, key, scale=1):
    value = row.get("timing", {}).get(key)
    if isinstance(value, dict):
        return f"{fmt(value['median'],scale)} [{fmt(value['min'],scale)}, {fmt(value['max'],scale)}]"
    return fmt(value, scale)


def table(columns, rows):
    return "\n".join(["| "+" | ".join(columns)+" |", "| "+" | ".join(["---"]*len(columns))+" |"]
                     + ["| "+" | ".join(map(str,r))+" |" for r in rows])


def figures(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    methods = ["adaptive", "ode", "fixed_cc41", "prepared16"]
    names = ["smooth_3d", "repeated_knot", "narrow_spans", "weight_1e12"]
    fig, axes = plt.subplots(2,2, figsize=(10,7), constrained_layout=True)
    for ax,name in zip(axes.flat,names):
        selected = [next((r for r in rows if r["kind"]=="length" and r["case"]==name
                         and r["method"]==m and r.get("comparison_tol")==1e-6),{}) for m in methods]
        times = [r.get("timing",{}).get("warm_s",{}).get("median",float("nan"))*1000 for r in selected]
        ax.bar(np.arange(4),times,color="#3376ad")
        ax.set(yscale="log", ylabel="Warm call (ms)",title=name.replace("_"," "))
        ax.set_xticks(np.arange(4),["Adaptive","ODE","CC41","Prepared"])
        right = ax.twinx()
        right.plot(np.arange(4),[max(r.get("abs_error",float("nan")),1e-15) for r in selected],"o--",color="#b24528")
        right.set(yscale="log",ylabel="Absolute reference error")
        ax.grid(axis="y",alpha=.2)
    fig.suptitle("Production arc-length APIs: execution cost and achieved accuracy\nBars: time; points: reference error, floored at 1e-15")
    fig.savefig(HERE/"length_comparison.png",dpi=150)
    plt.close(fig)
    maps = [r for r in rows if r["kind"]=="map" and "timing" in r]
    if maps:
        fig,axes = plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
        names = [r["case"].replace("_"," ")+f" / {r['comparison_tol']:g}" for r in maps]
        x = np.arange(len(maps))
        axes[0].bar(x,[r["timing"]["construction_s"] for r in maps],color="#3376ad")
        axes[0].set_ylabel("Cold construction (s)")
        axes[1].bar(x,[r["timing"]["warm_s"]["median"]*1000 for r in maps],color="#b24528")
        axes[1].set_ylabel("Warm 129-query batch (ms)")
        for ax in axes:
            ax.set_xticks(x,names,rotation=30,ha="right")
            ax.grid(axis="y",alpha=.2)
        fig.suptitle("Dense ODE map: construction and reuse")
        fig.savefig(HERE/"map_comparison.png",dpi=150)
        plt.close(fig)


def main():
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((HERE/"results").glob("*.json"))]
    rows = [r for r in rows if "kind" in r]
    figures(rows)
    sections = [r"""# Production JAX arc-length measurements

Measured on 27 October 2026 at revision `1dcc173`, before the API naming and
dispatcher cleanup. Timing values were not rerun for that API-only refactor;
the original measured-source fingerprints remain intact. API names below use
the current spelling. Every timed operation calls production
`nurbspy.jax`; there are no development integration/inverse implementations.
Raw JSON includes environment and measured-source fingerprints. Comparisons
are between production alternatives at that revision, not a re-created algorithm.

## Implemented behavior

- `get_arclength()` defaults to adaptive knot-aware quadrature, with separate
  `atol`/`rtol`, `tol=1e-6` convenience, direct/Leibniz differentiation and optional
  diagnostics. Scalar calls raise on unsuccessful status.
- `get_arclength()` dispatches statically to `get_arclength_quad_adaptive()`,
  `get_arclength_quad_fixed()` or `get_arclength_ode()`. Method-specific options
  are checked by the selected method. Fixed quadrature uses `n_points`, not a
  tolerance; the ambiguous former method name has been removed.
- `prepare_arclength()` supplies optional Gauss/basis tables for changing P/W;
  their resolution requires validation on the intended design family.
- Dense reciprocal ODE maps use independent normalized knot-span solves, exact
  endpoint anchors and sampled regularity/monotonicity checks. Queries perform
  interpolation only. No epsilon hides a reciprocal singularity.

Quadrature calculates total/partial length $S(u)=\int_{u_0}^u \|C'(t)\|\,dt$; the ODE
supplies repeated $u(s)$ queries. The map supports AD on regular curves and must
be rebuilt after geometry changes.

## Protocol

CPU float64, native-library thread limits, disabled persistent caches, serial
fresh workers and synchronized outputs. Imports, interpreter startup, curve
construction and independent references are outside kernel timings. Cold totals
include preparation, trace/lower, compilation and first execution; warm values
are medians [min,max]. One worker per cell does not capture all run-to-run
clock/compiler variation. Requested controls do not establish equal accuracy.

The length ODE allows 4,096 steps. Dense maps allow 512 steps per knot span and
force at least 16 per span; storage scales with span count. Map construction
includes quadrature, compilation, ODE solves and validation. Query compilation
is separate. Cached rebuilds reuse the same geometry and loaded code caches.

References use independent homogeneous SciPy splines and knot-split QUADPACK,
plus analytic cases. Round trips use independent partial lengths; unit speed
uses finite differences of returned maps. Knot-crossing stencils and low-speed
probes are excluded. Sampled validation is not a global certificate.
"""]
    lengths = [r for r in rows if r["kind"]=="length"]
    workflows = [r for r in rows if r["kind"]=="workflow"]
    maps = [r for r in rows if r["kind"]=="map"]
    sections.append("## Smooth length: cold stages and execution\n\nFirst and warm columns are milliseconds; other timing columns are seconds. Fixed rules have no requested tolerance.\n")
    sections.append(table(["Method / comparison control","Prep s","Trace s","Compile s","First ms","Cold total s","Warm ms [min,max]","Abs. error"],
        [(LABELS[r["method"]]+f" / {r['comparison_tol']:g}",timing(r,"preparation_s"),timing(r,"trace_lower_s"),timing(r,"compile_s"),timing(r,"first_execute_s",1000),timing(r,"cold_total_s"),timing(r,"warm_s",1000),fmt(r.get("abs_error"))) for r in lengths if r["case"]=="smooth_3d"]))
    sections.append("## Length robustness\n\nError / warm milliseconds at comparison control `1e-6`; unsuccessful status is retained.\n")
    matrix = []
    for name in sorted({r["case"] for r in lengths}):
        values = [name]
        for method in ["adaptive","ode","fixed_cc41","prepared16"]:
            r = next((r for r in lengths if r["case"]==name and r["method"]==method and r["comparison_tol"]==1e-6),None)
            values.append("—" if r is None else fmt(r.get("abs_error"))+" / "+timing(r,"warm_s",1000).split(" [")[0]+(" ("+r["status"]+")" if r["status"]!="ok" else ""))
        matrix.append(values)
    sections.append(table(["Case","Adaptive","ODE","Fixed CC41","Prepared GL16"],matrix))
    sections.append("\n![Length cost and accuracy](length_comparison.png)\n")
    sections.append("## Eight changing-design value/gradient evaluations\n\nBoth P and positive W change. Independent loss checks cover all eight designs outside timers; the base directional gradient uses an independent central difference.\n")
    sections.append(table(["Method","Cold total s","One warm ms","Eight designs ms [min,max]","Gradient rel. error","Max design loss error"],
        [(LABELS[r["method"]],timing(r,"cold_total_s"),timing(r,"warm_s",1000).split(" [")[0],timing(r,"eight_design_s",1000),fmt(r.get("gradient_relative_error")),fmt(r.get("design_max_loss_error"))) for r in workflows]))
    sections.append("## Dense ODE maps\n\nWarm batches contain 129 queries. Round-trip and unit-speed errors are independent of the total-length check.\n")
    sections.append(table(["Case / control","Construction s","Query trace s","Query compile s","First ms","Warm batch ms [min,max]","Cached rebuild+batch ms","Length error","Round-trip error","Unit-speed FD error","Status"],
        [(r["case"]+f" / {r['comparison_tol']:g}",timing(r,"construction_s"),timing(r,"trace_lower_s"),timing(r,"compile_s"),timing(r,"first_execute_s",1000),timing(r,"warm_s",1000),timing(r,"cached_build_query_s",1000).split(" [")[0],fmt(r.get("length_abs_error")),fmt(r.get("roundtrip_max_abs")),fmt(r.get("unit_speed_fd_error")),r["status"]) for r in maps]))
    sections.append("\n![Dense ODE construction and reuse](map_comparison.png)\n")
    failures = [r for r in rows if r["status"]!="ok"]
    sections.append("## Failures and explicit rejections\n")
    def reason(row):
        if row["status"] == "expected_rejection":
            return "Sampled stationary/flat speed; reciprocal ODE requires positive speed."
        return row.get("reason",str(row.get("diagnostics",{}))).replace("\n"," ").replace("|","/")[:600]
    sections.append(table(["Case","Method / control","Status","Reason/diagnostic"],[(r["case"],r["method"]+f" / {r['comparison_tol']:g}",r["status"],reason(r)) for r in failures]))
    sections.append(r"""## Numerical and maintenance limits

Adaptive quadrature is the general length method. Fixed/prepared rules are
useful for validated repeated-design families, but can badly underresolve
rational layers. Extreme weight failures remain visible; more workspace does
not cure floating-point resolution and error estimators are not certificates.

Dense queries remain interpolation-only. This regular-curve API rejects sampled
stationary speed and flat spans. An isolated stationary point can have a unique
continuous inverse with infinite derivative; a flat span has no unique ordinary
inverse. Those cases require a separately designed policy.

Near small speed, $|\Delta u|\lesssim\frac{|\Delta s|+\text{integration error}}{v_{\min}}$.
Local ODE controls do not certify global inverse or unit-speed error. Refine
tolerances/dense resolution for sensitive applications. Both geometry and query
sensitivities have analytic regressions; the norm's zero subgradient does not
make collapsed-curve length differentiable.

Retain this single production benchmark for future package/compiler maintenance.
Ordinary tests check values, interval contracts, failures, prepared tables,
independent map round trips/actual unit speed and AD, with no timing thresholds.
See the [usage guide](../../docs/source/getting_started/jax_arclength.md) and
[README](README.md) for assumptions and commands.
""")
    validation = HERE/"validation.json"
    if validation.exists():
        sections.append("## Validation\n\n"+table(["Check","Outcome"],json.loads(validation.read_text(encoding="utf-8")).items()))
    (HERE/"report.md").write_text("\n\n".join(sections)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
