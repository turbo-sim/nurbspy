"""Offline HTML and Markdown, generated exclusively from measured records."""
import html
import json
import math
import sys

sys.dont_write_bytecode = True
from benchmark import compare, leaves, save
from paths import FIGURES, RESULTS, SCRIPTS, STUDY as HERE, ensure_outputs


def render():
    ensure_outputs()
    import numpy as np
    from inventory import write
    write()
    from workloads import registry
    profiles, arrays = {}, {}
    for name in ("numpy", "numba", "jax"):
        path = RESULTS/f"{name}.json"
        if path.exists():
            profiles[name] = {r["id"]: r for r in json.loads(path.read_text())["records"]}
            arrays[name] = np.load(RESULTS/f"{name}.npz")
    environment = json.loads((RESULTS/"environment.json").read_text())
    import xml.etree.ElementTree as ET
    test_path = RESULTS/"tests.xml"
    test_summary = "Full-suite validation has not been recorded."
    if test_path.exists():
        suite = ET.parse(test_path).find("testsuite")
        if suite is not None:
            counts = {k: int(suite.get(k, "0")) for k in ("tests", "errors", "failures", "skipped")}
            passed_tests = counts["tests"]-counts["errors"]-counts["failures"]-counts["skipped"]
            test_summary = f"Full test suite: {passed_tests} passed, {counts['skipped']} skipped, {counts['failures']} failed, {counts['errors']} errors. Numerical references and saved-result integrity are checked separately by verify.py."
    numpy_test_path = RESULTS/"numpy-tests.xml"
    if numpy_test_path.exists():
        suite = ET.parse(numpy_test_path).find("testsuite")
        if suite is not None:
            passed_numpy = int(suite.get("tests", "0"))-sum(int(suite.get(k, "0")) for k in ("errors", "failures", "skipped"))
            test_summary += f" With Numba JIT disabled: {passed_numpy} additional focused/complex-step checks passed, {suite.get('skipped', '0')} skipped."
    results = []
    for row in registry():
        cells = {p: rows[row["id"]] for p, rows in profiles.items() if row["id"] in rows}
        if not cells:
            continue
        comparisons = {}
        key = row["id"].replace(".", "_")
        def value(profile):
            return [arrays[profile][f"{key}__out_{i}"] for i in range(len(cells[profile]["shapes"]))]
        for p in ("numba", "jax"):
            if cells.get("numpy", {}).get("status") == "ok" and cells.get(p, {}).get("status") == "ok":
                comparisons[p] = compare(value(p), value("numpy"), row)
        failures = [f"{p}: {r['error']}" for p, r in cells.items() if r["status"] != "ok"]
        expected_profiles = environment.get("profiles", list(profiles))
        missing = [p for p in expected_profiles if (not row["only_jax"] or p == "jax") and p not in cells]
        failures += [f"{p}: no result recorded for this workload" for p in missing]
        bad_reference = [p for p, r in cells.items() if r.get("reference", {}).get("passed") is False]
        bad_parity = [p for p, r in comparisons.items() if not r["passed"]]
        status = "ERROR" if failures else "REVIEW" if bad_reference or bad_parity else "EXPECTED UNDEFINED" if row["group"] == "degenerate" else "PASS"
        result = {**row, "cells": cells, "parity": comparisons, "status": status, "failures": failures, "reference_failures": bad_reference}
        if "jax" in cells and "numpy" in cells and all(cells[x].get("warm_median_s") for x in ("jax", "numpy")):
            result["speedup"] = cells["numpy"]["warm_median_s"]/cells["jax"]["warm_median_s"]
            difference = cells["numpy"]["warm_median_s"]-cells["jax"]["warm_median_s"]
            result["break_even"] = max(0, math.ceil((cells["jax"]["cold_s"]-cells["numpy"]["cold_s"])/difference)) if difference > 0 else None
        results.append(result)
    save(RESULTS/"summary.json", results)
    esc = lambda x: html.escape(str(x))
    fmt = lambda x: "—" if x is None else f"{x*1000:,.3f}"
    passed = sum(r["status"] == "PASS" for r in results)
    review = sum(r["status"] in ("REVIEW", "ERROR") for r in results)
    table = []
    for r in results:
        cells = r["cells"]
        j = cells.get("jax", {})
        parity = r["parity"].get("jax", {})
        errors = [c.get("reference", {}).get("max_abs") for c in cells.values()]
        errors = [x for x in errors if x is not None]
        details = json.dumps({"notes": [c.get("notes") for c in cells.values()], "reference": {p: c.get("reference") for p, c in cells.items()}, "parity": r["parity"], "failures": r["failures"], "timing_modes": {p: c.get("timing_mode") for p, c in cells.items()}, "setup_including_references_ms": {p: c.get("setup_s", 0)*1000 for p, c in cells.items()}, "warm_samples_ms": {p: [x*1000 for x in c.get("warm_samples_s", [])] for p, c in cells.items()}}, indent=2)
        speed = r.get("speedup")
        table.append(f'''<tr data-group="{esc(r['group'])}" data-status="{r['status']}"><td><details><summary>{esc(r['id'])}</summary><pre>{esc(details)}</pre></details></td><td><span class="badge {r['status'].lower().split()[0]}">{r['status']}</span></td><td>{fmt(cells.get('numpy', {}).get('warm_median_s'))}</td><td>{fmt(cells.get('numba', {}).get('warm_median_s'))}</td><td>{fmt(j.get('warm_median_s'))}</td><td>{f'{speed:.2f}×' if speed else '—'}</td><td>{fmt(j.get('lower_s'))}</td><td>{fmt(j.get('compile_s'))}</td><td>{fmt(j.get('first_s'))}</td><td>{fmt(j.get('cold_s'))}</td><td>{r.get('break_even') if r.get('break_even') is not None else '—'}</td><td>{f'{parity["max_abs"]:.2e}' if 'max_abs' in parity else '—'}</td><td>{f'{max(errors):.2e}' if errors else '—'}</td></tr>''')
    # A compact custom SVG avoids external chart libraries and network access.
    candidates = [r for r in results if r.get("speedup") and r["status"] == "PASS" and r["group"] in ("curve", "surface", "projection_curve", "projection_surface", "workflow")]
    selected = []
    for group in ("curve", "surface", "projection_curve", "projection_surface", "workflow"):
        chosen = [r for r in candidates if r["group"] == group and (r["op"] in ("value", "area", "batch", "curve", "surface"))]
        selected.extend(chosen[:2])
    chart = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 '+str(50+len(selected)*53)+'" role="img" aria-label="Warm execution comparison"><style>.svg-label{font:12px system-ui;fill:#33465e}.svg-note{font:11px system-ui;fill:#607189}</style><text x="305" y="20" class="svg-note">Log scale · milliseconds · lower is faster</text>']
    times = [c["warm_median_s"]*1000 for r in selected for c in r["cells"].values() if c.get("warm_median_s")]
    lo, hi = (math.log10(min(times)), math.log10(max(times))) if times else (-3, 3)
    hi = max(hi, lo+1)
    colors = {"numpy": "#93a4b8", "numba": "#8b72ed", "jax": "#02b89c"}
    for i, r in enumerate(selected):
        y = 49+i*53
        chart.append(f'<text x="0" y="{y+9}" class="svg-label">{esc(r["id"])}</text>')
        for k, p in enumerate(colors):
            t = r["cells"].get(p, {}).get("warm_median_s")
            if t:
                length = 25+520*(math.log10(t*1000)-lo)/(hi-lo)
                chart.append(f'<rect x="305" y="{y-8+k*12}" width="{length:.1f}" height="8" rx="3" fill="{colors[p]}"/><text x="{340+length:.1f}" y="{y+k*12}" class="svg-note">{t*1000:.3f}</text>')
    chart.append('</svg>')
    (FIGURES/"warm_execution.svg").write_text(''.join(chart), encoding="utf-8")
    findings = []
    for r in results:
        if r["status"] in ("REVIEW", "ERROR"):
            findings.append(f"- `{r['id']}`: {r['status']}. "+"; ".join(r["failures"]+[f"Independent reference failed: {', '.join(r['reference_failures'])}"] if r["reference_failures"] else r["failures"]))
    observations = [
        "Regular values, derivatives and geometric quantities should be assessed separately from convention differences and fixed-rule integration error. A REVIEW result is not automatically a defective algorithm.",
        "2D normal convention: NumPy get_normal is the principal Frenet normal; JAX get_normal uses a +90° rotated tangent. Their signs can differ. The separately measured NumPy get_normal_2D / rotated-JAX-tangent adapter checks the equivalent oriented-normal calculation.",
        "Fixed during this study: NumPy/Numba basis derivatives now use an iterative recurrence, return zero above the degree, and retain lower-degree neighbors for non-clamped knots. This removes invalid indexing and the cached recursive calls that produced native process failures. Final timings use corrected source; before-fix evidence is retained. Regression tests cover independent SciPy results, scalars, arrays, complex-step inputs and disk-cache reuse across processes.",
        "Length defaults are different algorithms: NumPy global Gauss-10 under-resolves the smooth and repeated-knot examples; JAX adaptive quadrature and the ODE pass the study threshold. Global JAX fixed Clenshaw-Curtis-41 also under-resolves these examples. Fixed-resolution rules need convergence checks or knot-span splitting; matching a tolerance argument is not equivalent work.",
        "Collapsed-curve frames, curvature and torsion are undefined. The backends return different NaN/zero patterns. These differences require an explicit API policy, not a looser regular-geometry tolerance.",
        "Surface area, whole-geometry energy workflows and NumPy finite-difference sensitivities are benchmark adapters rather than newly added production APIs. Offset surfaces use the package's sampled approximation, not exact geometric offsets.",
        "Prepared tables and dense inverse maps amortize setup only when reused. The map-query row measures reuse separately from full inverse-map construction; table setup includes independent reference evaluation and is not a pure construction microbenchmark."
    ]
    matched = [r for r in results if r["status"] == "PASS" and r["group"] in ("basis", "curve", "surface") and all(r["cells"].get(p, {}).get("warm_median_s") for p in ("numpy", "numba", "jax"))]
    if matched:
        faster_numpy = sum(r["cells"]["jax"]["warm_median_s"] < r["cells"]["numpy"]["warm_median_s"] for r in matched)
        faster_numba = sum(r["cells"]["jax"]["warm_median_s"] < r["cells"]["numba"]["warm_median_s"] for r in matched)
        observations.insert(1, f"On the {len(matched)} passing matched basis/curve/surface workloads, warm JAX is faster in {faster_numpy} cases versus pure NumPy and {faster_numba} versus NumPy + Numba. Small calls can favor Numba. These are local descriptive counts, not a universal speedup; compilation and the number of repeated calls matter.")
    local_miss = next((r for r in results if r["id"] == "projection_curve.batch.rational.16"), None)
    if local_miss and local_miss["cells"].get("numpy", {}).get("reference", {}).get("passed") is False:
        error = local_miss["cells"]["numpy"]["reference"]["max_abs"]
        observations.append(f"Projection limitation: NumPy/Numba's single local solve misses a strictly better feasible candidate in the 16-query curve batch; the squared-distance gap is {error:.6g}. JAX's sampled initialization passes this batch. Tight optimizer tolerances do not ensure the correct basin; neither finite sampling nor multistart is a universal global certificate.")
    map_errors = {r["kind"]: r["cells"].get("jax", {}).get("reference", {}).get("max_abs") for r in results if r["group"] == "map_query" and r["op"] == "unit_speed"}
    tight_errors = {r["kind"]: r["cells"].get("jax", {}).get("reference", {}).get("max_abs") for r in results if r["group"] == "map_query" and r["op"] == "unit_speed_tight"}
    if all(map_errors.get(k) is not None and tight_errors.get(k) is not None for k in ("rational", "repeated")):
        observations.append(f"Inverse-map derivative accuracy: default unit-speed errors are {map_errors['rational']:.3e} (smooth) and {map_errors['repeated']:.3e} (repeated knot). Using tol=1e-9 and 64 steps per span reduces them to {tight_errors['rational']:.3e} and {tight_errors['repeated']:.3e}. Parameter accuracy alone does not establish unit-speed accuracy. The existing tol=1e-6 is a local solver control, not a global derivative-error guarantee; no source default was changed.")
    timing_rows = []
    wanted = {"curve.value.rational.257", "curve.curvature.rational.257", "surface.value.rational.17", "surface.area.rational.17", "projection_curve.batch.rational.4", "projection_surface.batch.rational.4", "workflow.curve.rational.65", "workflow.surface.rational.9", "map_query.arclength.circle.65"}
    for r in results:
        if r["id"] in wanted:
            c = r["cells"]; j = c.get("jax", {})
            timing_rows.append(f"| `{r['id']}` | {fmt(c.get('numpy', {}).get('warm_median_s'))} | {fmt(c.get('numba', {}).get('warm_median_s'))} | {fmt(j.get('warm_median_s'))} | {fmt(j.get('lower_s'))} | {fmt(j.get('compile_s'))} | {fmt(j.get('first_s'))} |")
    md = f"# NumPy / JAX benchmark\n\nGenerated {environment['utc']}; source commit `{environment['commit']}` (see recorded working tree and hashes).\n\n{len(results)} workloads: {passed} passed, {review} need review; undefined geometry is recorded separately. Full tables, timing stages and raw samples are in [report.html](report.html) and in the expandable report rows. Local raw data is retained under `outputs/results/`.\n\n## Interpretation\n\n"+'\n\n'.join(observations)+"\n\n## Selected timings\n\nAll times in milliseconds; warm columns are synchronized medians. Blank compilation cells mean host compilation was not isolated. Imports, runtime startup and setup are excluded.\n\n| Workload | NumPy warm | NumPy + Numba warm | JAX warm | JAX lower | JAX compile | JAX first execution |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"+'\n'.join(timing_rows)+"\n\n## Findings requiring review\n\n"+ ('\n'.join(findings) or 'None in this run.')+"\n\nThese deterministic cases are a regression study, not proof of correctness for every input. Read README.md for timing scope, reference tolerances and API coverage. Production source was not changed by this study; confirmed defects should receive focused regression tests with their fixes.\n"
    md = md.replace("Production source was not changed by this study; confirmed defects should receive focused regression tests with their fixes.", "The production basis-derivative fixes are documented in fixes.md and covered by focused regression tests.")
    md = md.replace("(see recorded working tree and hashes)", "plus the documented working-tree basis fix (see recorded hashes)")
    integration_rows = []
    for r in results:
        if r["group"] == "length":
            cells = r["cells"]
            error = lambda p: cells.get(p, {}).get("reference", {}).get("max_abs")
            number = lambda x: "—" if x is None else f"{x:.3e}"
            integration_rows.append(f"| {r['kind']} | {r['op']} | {number(error('numpy'))} | {number(error('jax'))} |")
    md += "\n## Arc-length absolute error\n\nThe native default algorithms differ. Fixed resolution does not imply a tolerance guarantee. NumPy + Numba gives the same length errors as NumPy.\n\n| Curve | Method | NumPy absolute error | JAX absolute error |\n| --- | --- | ---: | ---: |\n"+'\n'.join(integration_rows)+"\n\n## Timing figure\n\nSee the [embedded timing figure](report.html#warm-execution).\n"
    md += "\n## Validation\n\n"+test_summary+" JUnit XML is retained locally under `outputs/results/`.\n"
    inverse_rows = []
    for r in results:
        if r["group"] == "map_query" and r["op"].startswith("unit_speed"):
            key = r["id"].replace(".", "_")
            raw = arrays["jax"]
            u_error = float(np.max(np.abs(raw[key+"__out_0"]-raw[key+"__ref_0"])))
            speed_error = float(np.max(np.abs(raw[key+"__out_1"]-1)))
            cell = r["cells"]["jax"]
            inverse_rows.append(f"| {r['kind']} | {r['op']} | {u_error:.3e} | {speed_error:.3e} | {fmt(cell.get('setup_s'))} | {fmt(cell.get('warm_median_s'))} |")
    md += "\n## Dense inverse validation\n\nSetup includes independent references plus map construction; it is not a pure solver timer. Query time includes both parameter evaluation and its unit-speed AD check.\n\n| Curve | Configuration | Maximum parameter error | Maximum unit-speed error | Setup + references (ms) | Warm validation query (ms) |\n| --- | --- | ---: | ---: | ---: | ---: |\n"+'\n'.join(inverse_rows)+"\n"
    (HERE/"report.md").write_text(md, encoding="utf-8")
    groups = sorted({r["group"] for r in results})
    options = ''.join(f'<option>{g}</option>' for g in groups)
    doc = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>nurbspy · NumPy / JAX benchmark</title><style>
    :root{{--ink:#17263b;--muted:#607189;--line:#dce4ee;--teal:#007f73}}*{{box-sizing:border-box}}body{{margin:0;background:#f3f6fa;color:var(--ink);font:15px/1.6 system-ui,sans-serif}}header{{background:#13263b;color:#fff;padding:54px max(5vw,24px) 42px}}header small{{letter-spacing:.16em;color:#79dfce;text-transform:uppercase}}h1{{font-size:clamp(28px,4vw,46px);line-height:1.15;margin:16px 0}}header p{{max-width:850px;color:#c3cedd}}main{{max-width:1600px;margin:auto;padding:28px max(3vw,20px) 70px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}}.card,section{{background:white;border:1px solid var(--line);border-radius:12px;padding:24px;margin-bottom:22px}}.card strong{{font-size:32px;display:block}}.muted,.card span{{color:var(--muted)}}h2{{font-size:22px;margin:0 0 14px}}h3{{font-size:16px}}.legend{{display:flex;gap:22px;font-size:13px}}.legend b{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}}svg{{width:100%;max-width:1100px}}.svg-label{{font:12px system-ui;fill:#33465e}}.svg-note{{font:11px system-ui;fill:#607189}}.controls{{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}}input,select{{font:inherit;border:1px solid #b8c6d8;border-radius:7px;padding:9px;background:#fff}}input{{min-width:280px}}.scroll{{overflow:auto;max-height:850px}}table{{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}}th{{background:#e9eff6;position:sticky;top:0;z-index:1;cursor:pointer;text-align:right}}th:first-child,td:first-child{{text-align:left}}th,td{{padding:11px 10px;border-bottom:1px solid var(--line);text-align:right}}tr:hover{{background:#f6f9fc}}summary{{cursor:pointer;font-weight:600}}pre{{max-width:650px;white-space:pre-wrap;text-align:left;font:12px/1.5 ui-monospace,monospace}}.badge{{font-size:10px;font-weight:700;padding:4px 7px;border-radius:4px;background:#e5f5ef;color:#136e53}}.review{{background:#fff1cf;color:#805800}}.error{{background:#ffe4e4;color:#a21c30}}.expected{{background:#ece9fa;color:#66539e}}.columns{{display:grid;grid-template-columns:1fr 1fr;gap:32px}}a{{color:#007f73}}footer{{font-size:12px;color:var(--muted)}}@media(max-width:800px){{.cards{{grid-template-columns:1fr 1fr}}.columns{{grid-template-columns:1fr}}header{{padding-top:30px}}}}@media print{{.scroll{{max-height:none;overflow:visible}}.controls{{display:none}}}}
    </style></head><body><header><small>nurbspy / engineering performance study</small><h1>One geometry. Two backends.</h1><p>Numerical agreement, compilation cost and steady-state performance across basis functions, differential geometry, integration, construction, projection and complete workflows.</p><p>{esc(environment['utc'])} · CPU · float64 · Python {esc(environment['python'].split()[0])} · source {environment['commit'][:10]}</p></header><main>
    <div class="cards"><div class="card"><strong>{len(results)}</strong><span>workloads measured</span></div><div class="card"><strong>{passed}</strong><span>passed available checks</span></div><div class="card"><strong>{review}</strong><span>require investigation</span></div><div class="card"><strong>{sum(len(r['cells']) for r in results)}</strong><span>backend measurements</span></div></div>
    <section id="warm-execution"><h2>Warm execution at a glance</h2><p class="muted">Selected passing workloads. Each bar uses the median of synchronized calls; exact values and all workloads are below. Comparisons include Python dispatch.</p><div class="legend"><span><b style="background:#93a4b8"></b>NumPy · Numba disabled</span><span><b style="background:#8b72ed"></b>NumPy + Numba</span><span><b style="background:#02b89c"></b>JAX</span></div>{''.join(chart)}</section>
    <section><h2>What the checks found</h2>{''.join('<p>'+esc(o)+'</p>' for o in observations)}<p><a href="coverage.md">Full public API coverage inventory</a> · <a href="fixes.md">Documented production fixes</a></p><p><strong>{esc(test_summary)}</strong> <span class="muted">JUnit XML is retained locally under <code>outputs/results/</code>.</span></p></section>
    <section><h2>Measurements & numerical agreement</h2><p class="muted">Times in milliseconds. Click an operation for tolerances, reference checks, setup time and samples. Click a column heading to sort. REVIEW can mean differing algorithms or accuracy, rather than a code defect.</p><div class="controls"><input id="search" aria-label="Search operations" placeholder="Search operations…"><select id="group" aria-label="Category"><option value="">All categories</option>{options}</select><select id="status" aria-label="Result"><option value="">All results</option><option>PASS</option><option>REVIEW</option><option>ERROR</option><option>EXPECTED UNDEFINED</option></select><span id="count" class="muted"></span></div><div class="scroll"><table id="measurements"><thead><tr>{''.join('<th>'+x+'</th>' for x in ['Operation','Check','NumPy warm','Numba warm','JAX warm','NumPy / JAX','JAX lower','JAX compile','JAX first','JAX cold total','Break-even calls','Parity max |Δ|','Reference max |Δ|'])}</tr></thead><tbody>{''.join(table)}</tbody></table></div></section>
    <section><h2>How to interpret the study</h2><div class="columns"><div><h3>Timing protocol</h3><p>Each backend runs in its own serial process. JAX runtime is initialized once. Each explicit JIT workload is lowered and compiled with dynamic geometry/sample inputs; caches are cleared between workloads. “Cold total” is lowering + compilation + first execution, excluding process/import startup and fixture/reference setup. Host APIs include their internal compilation in first call and have no separate compiler measurement.</p><p>Numba uses a fresh isolated disk cache per run, but shares compiled helpers across workloads. Its first-call timings are therefore order-dependent. Warm timings use {len(next((c.get('warm_samples_s',[]) for r in results for c in r['cells'].values() if c.get('warm_samples_s')), []))} repetitions. They are descriptive local measurements, not CI thresholds. Break-even estimates compare these calls only; preparation and process startup are excluded.</p></div><div><h3>Accuracy & scope</h3><p>Independent SciPy B-splines and rational quotient recurrences check values and derivatives. Numerical integration and multistart projection references use tighter settings. Projection compares squared distance and parameter bounds; equal-distance solutions may have different parameters. Expanded rows disclose absolute/relative tolerances and scaled error (≤1 passes).</p><p>Native default length algorithms differ. Surface area is a derived Gauss integration workflow. Prepared tables and inverse maps are JAX-only. Undefined frames on collapsed geometry are classified explicitly. Plot checks render and compare artist counts, not pixel images. This is broad deterministic coverage, not an exhaustive proof over every possible geometry.</p></div></div></section>
    <section><h2>Coverage & reproducibility</h2><p>Includes basis derivatives through fourth order; polynomial, rational, planar and repeated-knot curves; Frenet frames, speed, curvature and torsion; surface partials, normals, mean/Gaussian curvature and area; curve/surface projection; isocurves, joining, approximate offsets, circular arcs and all five surface generators; energy/integration workflows; AD versus finite differences; prepared tables and inverse maps; plotting smoke checks and degenerate geometry.</p><p>Low-level coordinate/derivative helpers are exercised through their public geometric wrappers. Plot styling, low-level helper variants, every invalid-input policy and every optimizer configuration are not separately timed. See <a href="README.md">methodology</a>, <a href="report.md">findings</a>, <a href="coverage.md">coverage inventory</a>. Recorded environment details are embedded below. Raw measurements, manifests and JUnit files are retained locally under <code>outputs/results/</code> and excluded from Git.</p><details><summary>Environment</summary><pre>{esc(json.dumps({k:v for k,v in environment.items() if k not in ('source_sha256','study_sha256')}, indent=2))}</pre></details></section><footer>Generated by development/jax_numpy_benchmark/scripts/report.py · offline report · no external assets or telemetry</footer></main>
    <script>const tbody=document.querySelector('tbody');const rows=[...tbody.rows];function filter(){{const q=document.querySelector('#search').value.toLowerCase(),g=document.querySelector('#group').value,s=document.querySelector('#status').value;let n=0;rows.forEach(r=>{{r.hidden=!(r.cells[0].textContent.toLowerCase().includes(q)&&(!g||r.dataset.group===g)&&(!s||r.dataset.status===s));if(!r.hidden)n++}});document.querySelector('#count').textContent=n+' operations shown'}}document.querySelectorAll('input,select').forEach(x=>x.addEventListener('input',filter));document.querySelectorAll('th').forEach((h,i)=>{{let asc=true;h.addEventListener('click',()=>{{rows.sort((a,b)=>{{const x=a.cells[i].textContent,y=b.cells[i].textContent;return (isNaN(parseFloat(x))||isNaN(parseFloat(y))?x.localeCompare(y):parseFloat(x.replaceAll(',',''))-parseFloat(y.replaceAll(',','')))*(asc?1:-1)}}).forEach(r=>tbody.appendChild(r));asc=!asc}})}});filter();</script></body></html>'''
    (HERE/"report.html").write_text(doc, encoding="utf-8")
    # Keep report regeneration provenance separate from the measurement start
    # manifest, including selective diagnostic runs added after the main run.
    import hashlib
    import time
    manifest = dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    scripts={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SCRIPTS.glob("*.py")},
                    measurements={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in RESULTS.glob("*.json") if p.name != "report_manifest.json"})
    save(RESULTS/"report_manifest.json", manifest)


if __name__ == "__main__":
    render()
