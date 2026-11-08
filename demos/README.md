# Nurbspy tutorials

These examples focus on defining geometry, computing useful quantities, and
showing the results through nurbspy's public APIs. Each script starts with a
description of what it illustrates.

After installing nurbspy, run a script directly from your IDE or terminal:

```powershell
poetry run python demos/00_basis_polynomials/01_basis_polynomials.py
poetry run python demos/01_curves/01_bezier_planar.py
poetry run python demos/02_surfaces/01_bilinear_planar.py
```

| Folder | Topics |
| --- | --- |
| [00_basis_polynomials](00_basis_polynomials/README.md) | Basis values and derivatives in Matplotlib |
| [01_curves](01_curves/README.md) | Bezier curves, weights, knots, arcs, derivatives, projection, merging |
| [02_surfaces](02_surfaces/README.md) | Patches, extrusion, ruling, Coons patches, revolution, derivatives, projection, STL |
| [03_viewer](03_viewer/README.md) | Editing, embedded viewers, and saving Figure state |

Curve and surface scripts use `nrb.Figure` and open one **editor** each. Planar
and spatial examples are separate scripts. Every example has a background grid
and the relevant control points, polygons/nets, frames, or boundaries. Display
sampling uses the default **Auto** quality, so no resolution value is needed.

`main(show=False)` returns one figure without opening a window. Teaching scripts
do not automatically generate images or HTML. The two STL examples write the
STL files they demonstrate.

Export all curve and surface examples as matching PNG and standalone HTML:

```powershell
python docs/export_demo_figures.py
```

Each pair goes under the example folder's `output/<script_name>/` directory.
See the [export instructions](../docs/EXPORT_DEMO_FIGURES.md). Gallery generation
stays separate from the teaching scripts.
