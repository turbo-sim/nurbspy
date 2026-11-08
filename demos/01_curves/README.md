# 01 - Curves

Run a script directly after installing nurbspy. Each script opens **one editor**
for one construction, with control points and a background grid. Planar and
spatial cases have separate scripts. Display quality uses the default **Auto**.

| Script | What it demonstrates |
| --- | --- |
| `01_bezier_planar.py` | Planar Bezier curve and control polygon |
| `02_bezier_spatial.py` | Spatial Bezier curve and Frenet-Serret frames |
| `03_rational_bezier_planar.py` | Planar curve with nonuniform weights |
| `04_rational_bezier_spatial.py` | Spatial curve with nonuniform weights |
| `05_bspline_planar.py` | Planar B-spline degree and clamped knots |
| `06_bspline_spatial.py` | Spatial B-spline and local frames |
| `07_nurbs_planar.py` | Planar rational B-spline |
| `08_nurbs_spatial.py` | Spatial rational B-spline |
| `09_circular_arc_planar.py` | Exact planar arc and analytical checks |
| `10_circular_arc_spatial.py` | Arc in an inclined plane and analytical checks |
| `11_derivatives_spatial.py` | Analytical derivatives against finite differences |
| `12_point_projection_planar.py` | Closest point on a curve |
| `13_merge.py` | Three connected segments joined into one curve |

`main(show=False)` returns one `nrb.Figure` without opening a window. Teaching
scripts do not automatically write PNG or HTML files. The STL examples write the
STL files whose export they demonstrate; these describe the initial construction.

For PNG **and** standalone interactive HTML, run from the repository root:

```powershell
python docs/export_demo_figures.py
```

The files are saved in this folder's `output/<script_name>/<script_name>.png`
and `.html`. Use `--only folder/script_name` to export a single example. See the
[export instructions](../../docs/EXPORT_DEMO_FIGURES.md).

Related boundary curves, isocurves, normals, and projection markers illustrate
the initial construction. They are independent snapshots, not constraints that
follow interactive edits. Curve Frenet frames and surface colour fields update
with the active edited geometry. Exported HTML supports viewing; use the Python
script for editing.
