"""Standalone reports for the five precomputed fitting formulations."""

from html import escape
from pathlib import Path
import json
import numpy as np
import nurbspy as nrb
from _experiment import reference_display


def rebuild_report(directory):
    """Re-export archived results without repeating any fitting calculations."""
    directory = Path(directory)
    observations = dict(np.load(directory / "observations.npz"))
    observations["sigma"] = float(observations["sigma"])
    rows = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    experiment = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    for row in rows:
        count = row["grid"]
        initial = nrb.NurbsSurface.from_yaml(directory / f"initial_{count:02d}.yaml")
        with np.load(directory / f"iterations_{count:02d}.npz") as saved:
            row["history"] = [
                dict(
                    iteration=int(iteration),
                    training_rms=float(rms),
                    surface=nrb.NurbsSurface(
                        points,
                        weights=weights,
                        u_degree=initial.p,
                        v_degree=initial.q,
                        u_knots=initial.U,
                        v_knots=initial.V,
                    ),
                )
                for iteration, rms, points, weights in zip(
                    saved["iteration"],
                    saved["training_rms"],
                    saved["control_points"],
                    saved["weights"],
                )
            ]
            for key in (
                "phase",
                "local_pass",
                "area",
                "constraint_violation",
                "outer_iteration",
                "seconds",
                "event_kind",
            ):
                if key in saved:
                    for frame, value in zip(row["history"], saved[key]):
                        if key in ("event_kind", "phase"):
                            frame[key] = str(value)
                        elif np.isfinite(value):
                            frame[key] = float(value)
        row["surface"] = nrb.NurbsSurface.from_yaml(directory / f"fit_{count:02d}.yaml")
    return export_report(
        rows,
        observations,
        directory,
        nonlinear=experiment["weight_bounds"] != [1.0, 1.0],
        study=experiment.get("study"),
    )


def export_report(records, observations, directory, *, nonlinear=False, study=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    study = study or (2 if nonlinear else 1)
    reference = reference_display()
    variants = {}
    cases = []
    for row in records:
        count = row["grid"]
        all_frames = row["history"]
        selected = np.unique(
            np.linspace(0, len(all_frames) - 1, min(61, len(all_frames)), dtype=int)
        )
        frames = [all_frames[i] for i in selected]
        entries = []
        for index, frame in enumerate(frames):
            name = f"{count}-{index}"
            model = frame["surface"]
            figure = nrb.Figure(
                model,
                name="Fitted surface",
                resolution=32,
                width=1000,
                height=750,
                display=dict(
                    control_points=False,
                    control_polygon=False,
                    grid=False,
                    opacity=0.92,
                ),
            )
            figure.title = "Surface viewer"
            figure.add(
                reference,
                name="Reference surface",
                visible=False,
                display=dict(
                    geometry_color="#b5c3cf",
                    opacity=0.3,
                    control_points=False,
                    control_polygon=False,
                    grid=False,
                    boundary=False,
                ),
            )
            figure.scatter(
                observations["points"].T, color="#263b4c", size=1.5, depth_test=True
            )
            figure.set_camera(
                position=[1.8, -1.6, 1.55],
                target=[0.5, 0.5, 0.0],
                up=[0, 0, 1],
                fov=35.0,
            )
            variants[name] = figure
            entries.append(
                dict(
                    id=name,
                    iteration=int(frame["iteration"]),
                    training_rms=frame["training_rms"],
                    constraint_violation=frame.get("constraint_violation"),
                    area=frame.get("area"),
                    outer_iteration=frame.get("outer_iteration"),
                    phase=frame.get("phase"),
                    local_pass=frame.get("local_pass"),
                    event_kind=frame.get("event_kind", "native_observation"),
                    seconds=frame.get("seconds"),
                )
            )
        summary = {
            key: value
            for key, value in row.items()
            if key not in ("surface", "history")
        }
        summary["frames"] = entries
        summary["recorded_frames"] = len(all_frames)
        cases.append(summary)
    viewer = next(iter(variants.values())).to_html(variants=variants)
    viewer = viewer.replace(
        "</head>",
        "<style>#toolbar{display:none}@media(max-width:600px){.orientation-widget{display:none!important}}</style></head>",
    )
    configuration = dict(
        cases=cases,
        nonlinear=nonlinear,
        samples=len(observations["points"]),
        sigma=observations["sigma"],
        study=study,
    )
    configuration_json = json.dumps(configuration, allow_nan=False).replace(
        "<", "\\u003c"
    )
    title = {
        1: "01 / B-spline surface fitting",
        2: "02 / Known-parameter rational fitting",
        3: "03 / Joint surface and correspondence fitting",
        4: "04 / Alternating projection and surface fitting",
        5: "05 / Constrained interior-point fitting",
    }[study]
    document = (
        _TEMPLATE.replace("@@TITLE@@", escape(title))
        .replace("@@VIEWER@@", escape(viewer, quote=True))
        .replace("@@DATA@@", configuration_json)
    )
    destination = directory / "report.html"
    destination.write_text(document, encoding="utf-8")
    if nonlinear:
        lines = [
            title.upper(),
            records[0]["solver"],
            "-------",
            f"{len(observations['points'])} shared observations. Reference errors are defined in FORMULATION.md.",
            f"Independent flat start for every grid; native tolerance {records[0].get('native_tolerance',records[0]['tolerance']):.0e}.",
            "Separate KKT checks disabled. Times are seconds, one measured run per grid.",
            "Setup includes compilation and warmup. Process startup and report export are excluded.",
            "",
            " Grid    Variables  Iterations  Compile     Warmup      Setup    Warm solve   Reference RMS   Fixed-weight RMS",
        ]
        for row in records:
            lines.append(
                f"{row['grid']:3d}x{row['grid']:<3d} {row['variables']:9d} {row['iterations']:11d} "
                f"{row['compile_seconds']:9.3f} {row['warmup_seconds']:10.3f} "
                f"{row['setup_seconds']:10.3f} {row['solve_seconds']:13.3f} "
                f"{row['clean_rms']:15.7g} {row.get('bspline_geometric_reference_rms',row['bspline_reference_rms']):18.7g}"
            )
        lines += [
            "",
            "WARM SOLVE COMPONENTS",
            "-------",
            " Grid     Value    Gradient    Hessian   Projection   Solver/interface remainder   Hessian entries",
        ]
        for row in records:
            remainder = (
                row["solve_seconds"]
                - sum(
                    row[k]
                    for k in (
                        "model_value_seconds",
                        "model_gradient_seconds",
                        "model_hessian_seconds",
                    )
                )
                - row.get("projection_seconds", 0.0)
            )
            lines.append(
                f"{row['grid']:3d}x{row['grid']:<3d} {row['model_value_seconds']:8.3f} {row['model_gradient_seconds']:11.3f} {row['model_hessian_seconds']:10.3f} {row.get('projection_seconds',0.0):12.3f} {remainder:28.3f} {row['hessian_nnz']:17d}"
            )
        lines += [
            "",
            "INTERPRETATION",
            "-------",
            "Kernel times synchronize results; the remainder is not a direct factorization measurement.",
            "Case 04 warm solve sums the surface-fitting IPM solves and library point projection.",
            "Use measured kernel shares to separate AD assembly costs from solver/interface costs.",
            "Compilation caching helps repeated setup, not an already warm solve.",
            "Playback contains sampled real observations and block endpoints, not interpolated iterates.",
            "Native convergence does not establish a global minimum or better reference-grid accuracy.",
            "Reference errors can exceed the fixed-weight baseline, especially with few samples per knot span.",
            "",
        ]
        (directory / "timings.txt").write_text("\n".join(lines), encoding="utf-8")
    print(
        f"Standalone report: {destination.resolve()} ({destination.stat().st_size/1024**2:.1f} MiB)",
        flush=True,
    )
    return destination


_TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>@@TITLE@@ | NurbsPy studies</title><style>
:root{--ink:#172e40;--muted:#617789;--blue:#087cbe;--line:#dce5ed;--paper:#f3f6fa;--green:#158568;--orange:#d67d28}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:var(--paper);font:14px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
header{padding:26px 38px 0;background:#fff;border-bottom:1px solid var(--line)}.brand{font-size:11px;font-weight:750;letter-spacing:.18em;color:var(--blue);text-transform:uppercase}
h1{margin:6px 0;font-size:29px;font-weight:650;letter-spacing:-.03em}header p{margin:0;color:var(--muted);max-width:950px}nav{display:flex;gap:26px;margin-top:20px}
nav a{color:var(--muted);text-decoration:none;padding:12px 0;border-bottom:3px solid transparent;font-weight:600}nav a.active{color:var(--blue);border-color:var(--blue)}
main{max-width:1680px;margin:24px auto;padding:0 28px 30px}.workspace{display:grid;grid-template-columns:270px minmax(0,1fr);gap:20px}.card{background:white;border:1px solid var(--line);border-radius:13px;box-shadow:0 4px 18px #17384a05}
aside{padding:22px;align-self:start}h2{font-size:17px;margin:0 0 12px}h3{font-size:12px;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);margin:22px 0 10px}
label.field{display:grid;gap:7px;margin:12px 0}select,button{font:inherit;border:1px solid #cbd8e3;border-radius:7px;background:white;color:var(--ink);padding:8px 10px}select{width:100%}button{cursor:pointer}button.primary{background:var(--blue);color:white;border-color:var(--blue)}button:disabled{opacity:.45;cursor:default}
.toggle{display:flex;align-items:center;gap:9px;margin:9px 0}.toggle input{accent-color:var(--blue)}.buttons{display:flex;gap:8px;flex-wrap:wrap}.note{font-size:12px;color:var(--muted)}
.viewer-card{overflow:hidden}.viewer-top{padding:13px 20px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:10px;align-items:center}.badge{font-size:11px;background:#e9f4fc;color:#086da6;padding:4px 9px;border-radius:20px}
iframe{display:block;width:100%;height:610px;border:0}.player{padding:16px 20px;border-top:1px solid var(--line)}.player input{width:100%;accent-color:var(--blue)}.player-head{display:flex;justify-content:space-between;align-items:center;gap:15px}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:18px}.metric{padding:15px 18px}.metric small{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.07em}.metric strong{font-size:21px;font-weight:600;font-variant-numeric:tabular-nums}.status{padding:14px 18px;margin-top:15px;font-size:12px}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:20px}.chart{padding:22px;min-width:0}.chart svg{width:100%;height:auto}.legend{font-size:12px;color:var(--muted)}.legend span{margin-right:20px}.table-card{padding:22px;margin-top:20px;overflow:auto}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;white-space:nowrap}th{text-align:right;font-size:11px;letter-spacing:.03em;color:var(--muted);background:#f6f9fc}td{text-align:right}th,td{padding:12px;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}tbody tr{cursor:pointer}tbody tr:hover{background:#edf7ff}.footnote{padding:18px 2px;color:var(--muted);max-width:1100px;font-size:12px}
[hidden]{display:none!important}#error{color:#b32628;white-space:pre-wrap}a{color:var(--blue)}
@media(max-width:1000px){.workspace{grid-template-columns:230px minmax(0,1fr)}.metrics{grid-template-columns:1fr 1fr}.charts{grid-template-columns:1fr}iframe{height:530px}}
@media(max-width:700px){header{padding:20px 20px 0}main{padding:0 12px}.workspace{grid-template-columns:1fr}aside{display:grid;grid-template-columns:1fr 1fr;gap:0 18px}.layers{grid-column:2;grid-row:1/5}.charts{grid-template-columns:1fr}iframe{height:420px}h1{font-size:24px}}
</style></head><body>
<header><div class="brand">NurbsPy / approximation studies</div><h1>@@TITLE@@</h1><p id="description"></p>
<nav><a href="#geometry" id="geometry-tab">Geometry explorer</a><a href="#refinement" id="refinement-tab">Grid refinement</a></nav></header>
<main><div id="error" role="alert"></div><section id="geometry"><div class="workspace"><aside class="card">
<h2>Explore the fit</h2><label class="field">Control grid<select id="grid" aria-label="Control grid"></select></label>
<div class="buttons"><button id="initial">Initial flat</button><button id="final" class="primary">Final fit</button></div>
<div class="layers"><h3>Visible layers</h3>
<label class="toggle"><input id="fit" type="checkbox" checked>Fitted surface</label>
<label class="toggle"><input id="reference" type="checkbox">Reference surface</label>
<label class="toggle"><input id="measurements" type="checkbox" checked>Target points</label>
<label class="toggle"><input id="controls" type="checkbox">Control points</label>
<label class="toggle"><input id="polygon" type="checkbox">Control polygon</label>
<p class="note">Rotate, pan and zoom in the viewer. Your camera and layer choices stay unchanged when switching results.</p></div>
</aside><div><div class="card viewer-card"><div class="viewer-top"><span id="scene-title">Preparing interactive viewer...</span><span class="badge">Precomputed / offline</span></div>
<iframe id="viewer" title="Interactive fitted surface" srcdoc="@@VIEWER@@"></iframe>
<div class="player"><div class="player-head"><span id="iteration"></span><span id="constraint-state" class="note"></span><div class="buttons"><button id="previous" aria-label="Previous iterate">Previous</button><button id="play">Play</button><button id="next" aria-label="Next iterate">Next</button></div></div>
<input id="timeline" type="range" min="0" value="0" aria-label="Optimization iterate"><p id="playback-note" class="note"></p></div></div>
<div class="metrics"><div class="card metric"><small>Training RMS / current state</small><strong id="train"></strong></div><div class="card metric"><small>Reference RMS / final fit</small><strong id="validation"></strong></div><div class="card metric"><small>Design variables</small><strong id="variables"></strong></div><div class="card metric"><small>Control points</small><strong id="cp"></strong></div></div>
<div id="status" class="card status"></div><div id="timing" class="card status" hidden></div></div></div></section>
<section id="refinement" hidden><div class="charts"><div class="card chart"><h2>Error versus control-grid size</h2><div id="grid-chart"></div><div class="legend"><span>Blue: training RMS</span><span>Orange: reference RMS</span></div></div><div class="card chart"><h2>Error versus design variables</h2><div id="variable-chart"></div><div class="legend"><span>Blue: training RMS</span><span>Orange: reference RMS</span></div></div></div>
<div class="card table-card"><h2>Grid-refinement results</h2><p class="note">Click a row or chart marker to inspect that result. <a href="#refinement" target="_blank" rel="noopener">Open this page in a new window</a></p><table><thead><tr><th>Control grid</th><th>Control points</th><th>Variables</th><th>Training RMS</th><th>Reference RMS</th><th class="case02">B-spline reference RMS</th><th>Observed order</th><th id="iterations-heading">Fitting IPM iterations</th><th class="case02">JAX compile</th><th class="case02">Setup total</th><th id="solve-heading">Warm solve</th><th>Status</th></tr></thead><tbody id="results"></tbody></table></div>
<div id="timing-panel" class="card table-card" hidden><h2>Compilation and warm solve</h2><p class="note">One sequential run per grid. Compilation includes tracing and compiling the value, gradient and sparse Hessian. Setup includes compilation, kernel warmup, fixed bases, sparsity/coloring and optional imports. Warm solve includes Pounce, its Python interface and observation recording; geometry export, final audits and HTML generation are excluded. JAX import is shared within each Python process; setup includes imports when needed. These are measured runs, not repeated timing averages.</p><div id="timing-chart"></div><h3>Measured solve components</h3><table><thead><tr><th>Control grid</th><th>Value</th><th>Gradient</th><th>Hessian</th><th>Library projection</th><th>Solver + interface remainder</th><th>Hessian entries</th><th>AD colors</th></tr></thead><tbody id="costs"></tbody></table><p class="note">Kernel times synchronize device results. Case 04 includes library projection as a separate component. The remainder includes native factorization, globalization, bound handling, array validation/copying and reporting; it is not a direct measurement of factorization alone. No separate KKT solver checks are performed; explicit block and final geometry audits are outside warm timing.</p></div>
<p class="footnote" id="error-note">RMS is Euclidean 3D correspondence error at known parameters, in normalized coordinate units. Reference RMS uses the analytic function on a separate 193 x 193 grid. Observed order uses h = 1 / (n - 3), the cubic knot-span spacing. These control grids are not nested; monotone error reduction is measured, not assumed. Both chart axes are logarithmic. This page shows spatial grid refinement, not optimizer convergence.</p></section>
<p class="footnote" id="method-note"></p></main>
<script>
const data=@@DATA@@,$=id=>document.getElementById(id),num=x=>Number(x).toExponential(3);
let selected=0,step=0,playing=false,ready=false,serial=Promise.resolve();
const cases=data.cases,viewer=$('viewer');
if(!data.nonlinear){document.querySelectorAll('th.case02').forEach(node=>node.remove());$('solve-heading').textContent='Solve time';$('iterations-heading').textContent='Iterations';}
const seconds=x=>x==null?'--':`${Number(x).toFixed(3)} s`;
const descriptions={
1:'Known point parameters; unit weights; direct tensor-product linear least squares.',
2:'Known, fixed u/v labels; control coordinates and positive weights optimized by exact sparse-Hessian interior point.',
3:'Unlabeled cloud; control coordinates, weights and all point parameters optimized jointly by exact sparse-Hessian interior point.',
4:'Unlabeled cloud; alternating point projection and surface optimization. Surface fitting uses exact sparse-Hessian interior point; projection reuses the NurbsPy bounded Newton API.',
5:'Known point parameters; bounded heights/interior weights, fixed planar controls and boundary weights, with area/orientation constraints.'};
$('description').textContent=`${descriptions[data.study]} ${data.samples.toLocaleString()} shared target points.`;
$('method-note').textContent=`Independent flat initial surfaces, cubic B-splines and the same analytic six-hill/valley terrain. Noise standard deviation: ${data.sigma}. Nonlinear surface fits use Pounce interior point and exact sparse Hessians. No Python server or internet connection is needed.`;
if(data.study===3)$('method-note').textContent+=' Local parameter windows make each Hessian sparse; windows are recentered when their artificial-bound multipliers prevent stationarity on the global [0,1] domain. Every IPM pass updates all variables jointly. No projection/refit alternation is used.';
if(data.study===4)$('method-note').textContent+=' The objective, geometric bounds and regularization match case 03. Compiled fitting and projection kernels are reused between cycles. Convergence requires both blocks to be stationary; exhausting the cycle budget is labeled Not converged.';
if(data.study===5)$('method-note').textContent+=' Pounce receives a sparse constraint Jacobian and the full multiplier-weighted Lagrangian Hessian. Area is limited to 1+0.85*(target area-1); sampled planar Jacobians must exceed 0.05. Finer quadrature and a denser guard grid audit the result.';
const geometric=data.study===3||data.study===4;
$('error-note').textContent=geometric?'Training RMS uses optimized point parameters. Reference RMS uses the existing NurbsPy local projection API on a separate 65 x 65 analytic grid. These are one-sided local distances, not certified global nearest distances. The fixed-label 193 x 193 score is saved as parametric_reference_rms. The baseline column uses the same geometric reference metric.':'Training RMS is Euclidean 3D error at supplied fixed parameter labels. Reference RMS compares the fit and analytic terrain at the same labels on an independent 193 x 193 grid; no projection is used.';
$('error-note').textContent+=' Observed order uses h=1/(n-3) and is shown only between converged fits. The grids are not nested. Nonlinear native convergence does not guarantee a global minimum or lower reference error. The reference surface in the viewer is a display approximation.';
$('playback-note').textContent=data.nonlinear?'Stored native observations and block endpoints, sampled to at most 61 snapshots per grid. Native coordinates are last objective evaluations with unknown acceptance. Iteration numbers are preserved; no interpolated or fabricated frames are added. Full metrics and histories are archived beside this report.':'One direct linear solve per grid. Initial/final views are a comparison, not iterative optimization.';
$('timing-panel').hidden=!data.nonlinear;
for(const row of cases)$('grid').add(new Option(`${row.grid} x ${row.grid}`,row.grid));
function route(){const refinement=location.hash==='#refinement';$('geometry').hidden=refinement;$('refinement').hidden=!refinement;$('geometry-tab').classList.toggle('active',!refinement);$('refinement-tab').classList.toggle('active',refinement);if(refinement)stop();}
window.addEventListener('hashchange',route);route();
function stop(){playing=false;$('play').textContent='Play';}
async function show(){
    if(!ready)return;const row=cases[selected],frame=row.frames[step];
    await viewer.contentWindow.nurbspySelectVariant(frame.id);
    $('scene-title').textContent=`${row.grid} x ${row.grid} control grid / ${data.nonlinear?'rational NURBS':'B-spline'}`;
    $('timeline').max=row.frames.length-1;$('timeline').value=step;
    $('constraint-state').textContent=frame.constraint_violation==null?'':`Violation ${num(frame.constraint_violation)} | Area ${frame.area.toFixed(5)} / limit ${row.area_limit.toFixed(5)}`;
    const labels={initial:'Initial flat surface',solution:'Final returned solution',native_observation:'Native observation',projection:'Projected correspondences',surface_solution:'Surface-step solution',local_solution:'Local joint solution',sparsity_restart:'Recentered parameter windows',alternating_restart:'Surface-step start'};
    const phase=frame.outer_iteration?` / cycle ${frame.outer_iteration}`:(frame.local_pass?` / local pass ${frame.local_pass}`:'');
    $('iteration').textContent=data.nonlinear?`${labels[frame.event_kind]||frame.event_kind} / IPM iteration ${frame.iteration}${phase}`:(step===0?'Initial flat surface':'Direct least-squares solution');
    $('train').textContent=num(frame.training_rms);$('validation').textContent=num(row.clean_rms);$('variables').textContent=row.variables.toLocaleString();$('cp').textContent=(row.grid**2).toLocaleString();
    $('status').textContent=`${row.success?'Completed':'Stopped before convergence'}: ${row.message} | ${row.solve_seconds.toFixed(2)} s | ${row.nfev} function evaluations${row.reference_projection_success===false?' | Reference projection stopped before convergence':''}`;
    $('timing').hidden=!data.nonlinear;
    $('timing').textContent=`JAX compilation ${seconds(row.compile_seconds)} | Kernel warmup ${seconds(row.warmup_seconds)} | Setup total ${seconds(row.setup_seconds)} | Warm solve ${seconds(row.solve_seconds)} | ${row.nhev} Hessian evaluations`;
    $('previous').disabled=step===0;$('next').disabled=step===row.frames.length-1;
    $('play').hidden=!data.nonlinear;
}
function request(){serial=serial.then(show).catch(e=>{$('error').textContent=e.message;stop();});return serial;}
$('grid').onchange=()=>{stop();selected=cases.findIndex(x=>x.grid===Number($('grid').value));step=cases[selected].frames.length-1;request();};
$('initial').onclick=()=>{stop();step=0;request();};$('final').onclick=()=>{stop();step=cases[selected].frames.length-1;request();};
$('previous').onclick=()=>{stop();step=Math.max(0,step-1);request();};$('next').onclick=()=>{stop();step=Math.min(cases[selected].frames.length-1,step+1);request();};
$('timeline').oninput=()=>{stop();step=Number($('timeline').value);request();};
$('play').onclick=async()=>{if(playing){stop();return;}playing=true;$('play').textContent='Pause';if(step===cases[selected].frames.length-1)step=0;while(playing){await request();await new Promise(r=>setTimeout(r,250));if(!playing)break;if(step===cases[selected].frames.length-1){stop();break;}step++;}};
async function layers(){if(!ready)return;await viewer.contentWindow.nurbspySetLayers({objects:{'Fitted surface':{visible:$('fit').checked,control_points:$('controls').checked,control_polygon:$('polygon').checked},'Reference surface':{visible:$('reference').checked}},markers:$('measurements').checked});}
for(const id of ['fit','reference','measurements','controls','polygon'])$(id).onchange=()=>{serial=serial.then(layers).catch(e=>$('error').textContent=e.message);};
const timer=setInterval(()=>{try{if(viewer.contentWindow.nurbspyReportReady){clearInterval(timer);ready=true;step=cases[0].frames.length-1;request();}}catch(e){}},100);
function choose(i){selected=i;$('grid').value=cases[i].grid;step=cases[i].frames.length-1;location.hash='geometry';request();}
function chart(element,key){const w=600,h=330,l=72,r=25,t=25,b=60;const xs=cases.map(x=>x[key]),ys=cases.flatMap(x=>[x.training_rms,x.clean_rms]).map(x=>Math.max(x,1e-16));let xmin=Math.log10(Math.min(...xs)),xmax=Math.log10(Math.max(...xs));if(xmax===xmin)xmax=xmin+1;let ymin=Math.floor(Math.log10(Math.min(...ys))),ymax=Math.ceil(Math.log10(Math.max(...ys)));if(ymax===ymin)ymax=ymin+1;
const X=x=>l+(Math.log10(x)-xmin)/(xmax-xmin)*(w-l-r),Y=y=>h-b-(Math.log10(Math.max(y,1e-16))-ymin)/(ymax-ymin)*(h-t-b);
let svg=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Logarithmic grid-refinement error chart">`;
for(let p=ymin;p<=ymax;p++){const y=Y(10**p);svg+=`<line x1="${l}" x2="${w-r}" y1="${y}" y2="${y}" stroke="#e3eaf1"/><text x="${l-10}" y="${y+4}" text-anchor="end" fill="#617789" font-size="12">1e${p}</text>`;}
let lastLabel=-Infinity;for(const x of xs){if(X(x)-lastLabel<64)continue;lastLabel=X(x);svg+=`<text x="${X(x)}" y="${h-b+23}" text-anchor="middle" fill="#617789" font-size="12">${x.toLocaleString()}</text>`;}
for(const [metric,color] of [['training_rms','#087cbe'],['clean_rms','#d67d28']]){svg+=`<polyline points="${cases.map(x=>`${X(x[key])},${Y(x[metric])}`).join(' ')}" fill="none" stroke="${color}" stroke-width="2.5"/>`;cases.forEach((row,i)=>{svg+=`<circle data-case="${i}" cx="${X(row[key])}" cy="${Y(row[metric])}" r="5" fill="white" stroke="${color}" stroke-width="2.5" tabindex="0" style="cursor:pointer"><title>${row.grid} x ${row.grid}: ${metric} = ${num(row[metric])}</title></circle>`;});}
svg+=`<text x="${w/2}" y="${h-8}" text-anchor="middle" font-size="12" fill="#617789">${key==='grid'?'Control points per direction':'Number of design variables'}</text></svg>`;$(element).innerHTML=svg;$(element).querySelectorAll('[data-case]').forEach(p=>{p.onclick=()=>choose(Number(p.dataset.case));p.onkeydown=e=>{if(e.key==='Enter')p.click();};});}
chart('grid-chart','grid');chart('variable-chart','variables');
cases.forEach((row,i)=>{const prev=cases[i-1],order=prev&&prev.success&&row.success&&prev.reference_projection_success!==false&&row.reference_projection_success!==false?Math.log(prev.clean_rms/row.clean_rms)/Math.log((row.grid-3)/(prev.grid-3)):null;const tr=document.createElement('tr');tr.innerHTML=`<td>${row.grid} x ${row.grid}</td><td>${row.grid**2}</td><td>${row.variables.toLocaleString()}</td><td>${num(row.training_rms)}</td><td>${num(row.clean_rms)}</td>${data.nonlinear?`<td>${num(geometric?row.bspline_geometric_reference_rms:row.bspline_reference_rms)}</td>`:''}<td>${order===null?'--':order.toFixed(2)}</td><td>${row.iterations}</td>${data.nonlinear?`<td>${seconds(row.compile_seconds)}</td><td>${seconds(row.setup_seconds)}</td>`:''}<td>${seconds(row.solve_seconds)}</td><td>${row.success?'Completed':'Not converged'}${row.reference_projection_success===false?' / reference projection incomplete':''}</td>`;tr.onclick=()=>choose(i);$('results').append(tr);});
if(data.nonlinear){
    const w=1000,rowHeight=48,left=110,right=80,top=22,h=top+cases.length*rowHeight+40,max=Math.max(...cases.map(r=>r.setup_seconds+r.solve_seconds)),X=x=>(w-left-right)*x/max;
    let svg=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Setup and warm solve times by control grid"><text x="${left}" y="14" font-size="12" fill="#617789">Blue: JAX compilation / Light blue: other setup and warmup / Orange: warm solve</text>`;
    cases.forEach((r,i)=>{const y=top+i*rowHeight,compile=r.compile_seconds,other=Math.max(0,r.setup_seconds-compile);svg+=`<text x="${left-15}" y="${y+24}" text-anchor="end" font-size="13" fill="#172e40">${r.grid} x ${r.grid}</text><rect x="${left}" y="${y+7}" width="${X(compile)}" height="25" fill="#087cbe"/><rect x="${left+X(compile)}" y="${y+7}" width="${X(other)}" height="25" fill="#b4d9ef"/><rect x="${left+X(r.setup_seconds)}" y="${y+7}" width="${X(r.solve_seconds)}" height="25" fill="#d67d28"/><text x="${left+X(r.setup_seconds+r.solve_seconds)+8}" y="${y+24}" font-size="12" fill="#617789">${(r.setup_seconds+r.solve_seconds).toFixed(2)} s</text>`;
        const remainder=Math.max(0,r.solve_seconds-r.model_value_seconds-r.model_gradient_seconds-r.model_hessian_seconds-(r.projection_seconds||0)),tr=document.createElement('tr');tr.innerHTML=`<td>${r.grid} x ${r.grid}</td><td>${seconds(r.model_value_seconds)}</td><td>${seconds(r.model_gradient_seconds)}</td><td>${seconds(r.model_hessian_seconds)}</td><td>${seconds(r.projection_seconds||0)}</td><td>${seconds(remainder)}</td><td>${r.hessian_nnz.toLocaleString()}</td><td>${r.hessian_colors}</td>`;$('costs').append(tr);
    });$('timing-chart').innerHTML=svg+'</svg>';
}
</script></body></html>"""
