"""Regenerate all five studies sequentially, then write a local report index."""

from html import escape
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DEMOS = (
    ("demo_01_linear_least_squares.py", "01_linear", "Linear B-spline / known u/v"),
    (
        "demo_02_nonlinear_weights.py",
        "02_known_parameters",
        "Rational surface / known u/v",
    ),
    (
        "demo_03_joint_parameters.py",
        "03_joint_parameters",
        "Joint surface, weights and u/v",
    ),
    (
        "demo_04_alternating_projection.py",
        "04_alternating_projection",
        "Alternating projection and fitting",
    ),
    (
        "demo_05_constrained_ipm.py",
        "05_constrained",
        "Constrained interior-point fitting",
    ),
)


def write_index():
    """Link the self-contained reports and summarize their actual convergence."""
    rows = []
    for _, folder, title in DEMOS:
        records = json.loads((ROOT / "output" / folder / "metrics.json").read_text())
        converged = sum(bool(r["success"]) for r in records)
        rows.append(
            f'<tr><td><a href="{folder}/report.html">{escape(title)}</a></td>'
            f"<td>4, 8, 16, 32</td><td>{converged}/{len(records)}</td>"
            f'<td>{sum(r["solve_seconds"] for r in records):.3f} s</td></tr>'
        )
    document = (
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NurbsPy / surface-fitting studies</title><style>
body{margin:60px auto;max-width:1100px;padding:0 24px;background:#f3f6fa;color:#172e40;font:15px/1.6 system-ui}
h1{font-size:32px;letter-spacing:-.03em}small{color:#087cbe;letter-spacing:.15em}
main{background:white;border:1px solid #dce5ed;border-radius:14px;padding:30px;overflow:auto}
table{width:100%;border-collapse:collapse;white-space:nowrap}th,td{text-align:left;padding:16px 12px;border-bottom:1px solid #dce5ed}th{font-size:12px;color:#617789}a{color:#087cbe}
</style><small>NURBSPY / APPROXIMATION STUDIES</small><h1>Surface fitting: five formulations</h1>
<p>One analytic terrain, 2,025 observations, independent flat starts and identical refinement grids.
Cases 02-05 fit with Pounce interior point and exact sparse Hessians.
Point projection reuses the existing NurbsPy JAX bounded-Newton API.</p><main>
<table><thead><tr><th>Standalone report</th><th>Controls per direction</th><th>Converged fits</th><th>Total warm solve</th></tr></thead><tbody>"""
        + "".join(rows)
        + """</tbody></table></main>
<p>Each report includes geometry exploration, stored optimization snapshots, refinement errors and timings.
Convergence labels preserve native and formulation-specific checks; a budget exit is not success.
Case 04 sums the warm projection and surface solves. The reports work offline.</p></html>"""
    )
    (ROOT / "output" / "index.html").write_text(document, encoding="utf-8")


def main():
    """Run independent Python processes to separate each demo's setup and timings."""
    for script, _, title in DEMOS:
        print(f"\n{'='*65}\n{title}\n{'='*65}", flush=True)
        subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)
    write_index()
    print(f"\nAll reports: {ROOT/'output/index.html'}", flush=True)


if __name__ == "__main__":
    main()
