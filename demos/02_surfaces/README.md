# 02 — Surfaces

Run from the repository root with `python demos/02_surfaces/<script_name>.py`.
Every geometry view uses Three.js through `nrb.Figure`.

| Script | What it demonstrates |
| --- | --- |
| `01_bilinear` | Four-corner planar and spatial patches |
| `02_bezier` | A bicubic control net |
| `03_rational_bezier` | Weighted surface control points |
| `04_bspline` | Mean-curvature colouring, normals, and exact isocurves |
| `05_nurbs` | Rational surface degrees, knots, and weights |
| `06_extruded` | A generating curve and extrusion, including a cylinder |
| `07_ruled` | Surfaces spanned by straight lines between boundary curves |
| `08_coons` | A patch interpolating four boundary curves |
| `09_annular_coons` | An annular patch between circular boundaries |
| `10_oloid_ruled` | An approximate oloid assembled from two ruled patches |
| `11_revolution` | Revolution surfaces and Gaussian-curvature colouring |
| `12_derivatives` | Native partial derivatives checked by finite differences |
| `13_point_projection` | Closest surface parameters and a projection segment |
| `14_merge` | Joining compatible patches in either parameter direction |
| `15_stl_printing` | Closed STL export and the actual reloaded triangle meshes |

Each script defines its geometry and opens the views directly through
`nrb.Figure`. Call `main(show=False)` to reuse the returned figures without
opening windows. No images, scene files, or reports are saved automatically.
The STL tutorial in the surfaces folder writes the STL files it demonstrates.

See the [Figure guide](../../docs/source/figures.md) for standalone interactive
HTML files and PNG export.
