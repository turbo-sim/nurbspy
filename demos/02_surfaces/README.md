# 02 - Surfaces

Run a script directly after installing nurbspy. Each script opens **one editor**
for one construction, with control points and a background grid. Planar and
spatial cases have separate scripts. Display quality uses the default **Auto**.

| Script | What it demonstrates |
| --- | --- |
| `01_bilinear_planar.py` | Planar four-corner patch |
| `02_bilinear_spatial.py` | Warped four-corner patch |
| `03_bezier.py` | Polynomial Bezier control net |
| `04_rational_bezier.py` | Nonuniform control-point weights |
| `05_bspline.py` | Mean curvature, exact isocurves, and initial normals |
| `06_nurbs.py` | Rational surface degrees, knots, and weights |
| `07_extruded_profile.py` | Extrusion of a weighted profile |
| `08_extruded_cylinder.py` | Extrusion of an exact circle |
| `09_ruled_profiles.py` | Straight rulings between two weighted profiles |
| `10_ruled_circles.py` | Straight rulings between crossed circles |
| `11_coons.py` | Interpolation of four boundary curves |
| `12_annular_coons.py` | Annular patch between circular boundaries |
| `13_oloid_ruled.py` | One oloid assembled from two ruled patches |
| `14_revolved_profile.py` | Profile revolved about an inclined axis |
| `15_revolved_sphere.py` | Sphere construction and curvature checks |
| `16_revolved_torus.py` | Torus with signed Gaussian curvature |
| `17_derivatives.py` | Partial derivatives against finite differences |
| `18_point_projection.py` | Closest point on a surface |
| `19_merge_u.py` | One joined patch in the u direction |
| `20_merge_v.py` | One joined patch in the v direction |
| `21_stl_torus.py` | Exact torus, closed STL export, and mesh validation |
| `22_stl_oloid.py` | Two-patch oloid, closed STL export, and mesh validation |

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

Surface isocurves are linked display overlays: they follow point and weight
edits, dragging previews, and undo/redo. Toggle **Isocurves** under Display > Geometry.
The oloid rulings are linked separately to each patch. These lines are not
independent objects in the editor's Object selector.

Generating boundary curves, manually drawn normals, and projection markers
illustrate the initial construction as independent snapshots. Curve Frenet frames
and surface colour fields also update with the active edited geometry. Exported
HTML supports viewing; use the Python script for editing.
