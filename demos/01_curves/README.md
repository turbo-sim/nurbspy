# 01 — Curves

Run from the repository root with `python demos/01_curves/<script_name>.py`.
Every geometry view uses Three.js through `nrb.Figure`.
Only construction views are shown, with control points and polygons. The
Bézier, B-spline, NURBS, and derivative examples also show Frenet–Serret frames.
Spatial views include a reference grid; planar views keep a plain background.

| Script | What it demonstrates |
| --- | --- |
| `01_bezier` | Planar/spatial Bezier curves and control polygons |
| `02_rational_bezier` | The effect of nonuniform weights |
| `03_bspline` | Degree and clamped knot vectors |
| `04_nurbs` | Rational B-splines, control polygons, and Frenet frames |
| `05_circular_arc` | Exact rational circular arcs and analytical checks |
| `06_derivatives` | Native derivatives against finite differences |
| `07_point_projection` | A point, its closest curve point, and the projection segment |
| `08_merge` | Compatible segments and the merged curve |

Each script defines its geometry and opens the views directly through
`nrb.Figure`. Call `main(show=False)` to reuse the returned figures without
opening windows. No images, scene files, or reports are saved automatically.
The STL tutorial in the surfaces folder writes the STL files it demonstrates.

See the [Figure guide](../../docs/source/figures.md) for standalone interactive
HTML files and PNG export.
