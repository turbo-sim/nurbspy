# Nurbspy tutorials

These examples focus on defining geometry, computing useful quantities, and
showing the results through nurbspy's public APIs. Each script starts with a
description of what it illustrates.

After installing nurbspy, run a script directly from your IDE or terminal:

```powershell
poetry run python demos/00_basis_polynomials/01_basis_polynomials.py
poetry run python demos/01_curves/01_bezier.py
poetry run python demos/02_surfaces/01_bilinear.py
```

| Folder | Topics |
| --- | --- |
| [00_basis_polynomials](00_basis_polynomials/README.md) | Basis values and derivatives in Matplotlib |
| [01_curves](01_curves/README.md) | Bezier curves, weights, knots, arcs, derivatives, projection, merging |
| [02_surfaces](02_surfaces/README.md) | Patches, extrusion, ruling, Coons patches, revolution, derivatives, projection, STL |
| [03_viewer](03_viewer/README.md) | Editing, embedded viewers, and saving Figure state |

Curve and surface examples use `nrb.Figure`. Each geometry case has one view
with the relevant control points, polygons/nets, frames, boundaries, or parameter
curves. Spatial views include a reference grid. Display sampling uses the default
**Auto** quality, so the scripts do not specify resolution values. Use Display's
**Fast** or **Fine** setting if you want to change the detail/speed tradeoff.

`main(show=False)` builds and returns the figures without opening windows, so
other Python code can reuse them. The examples do not automatically generate
images, reports, or data archives. The STL tutorial writes STL files because
exporting a printable mesh is the capability it teaches.

Interactive HTML and image export are described in the
[Figure guide](../docs/source/figures.md). Presentation and publication tooling
belongs outside these teaching scripts.
