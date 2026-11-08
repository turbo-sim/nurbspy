# Shared viewer and editor examples

Run from the main checkout. Geometry uses Point inputs; arrays remain supported.
All settings are readable values in the scripts, with no command-line options.
The default appearance is solid colour; diagnostics are available in Display.

```powershell
poetry install
poetry run python demos/03_viewer/01_curve.py
poetry run python demos/03_viewer/02_surface.py
```

| Geometry | Desktop | Embedded `GEOMETRY` setting |
| --- | --- | --- |
| Curve | [01_curve.py](01_curve.py) | `"curve"` |
| Surface | [02_surface.py](02_surface.py) | `"surface"` |

Each script has global `EDITABLE` and `DISPLAY` settings.
`EDITABLE = True` opens the editor; set it to `False` for the read-only viewer.
The embedded demo also exposes `GEOMETRY` and `PORT`. Geometry and display settings are shared
between modes within each demo, so there is no separate viewer/editor file.

Display quality defaults to **Auto** in both modes. The viewer chooses sampling
from the geometry, reuses a coarse grid while dragging, and refines after release.
Display offers **Fast** and **Fine** alternatives; no numeric resolution is needed.
STL sampling and PNG pixel dimensions are separate export settings.

Desktop uses a native webview with no HTTP server. Apply returns a new NumPy
geometry, Cancel returns None, and original input remains unchanged. Save the
returned object with `to_yaml()` if desired. Load with the appropriate NumPy/JAX
class's `from_yaml()` before viewing. Editors support Shift-click/Shift-drag
multi-selection and group dragging. Undo/Redo buttons and Ctrl+Z/Ctrl+Shift+Z
(Command on macOS) restore completed geometry edits. File dialogs are omitted.

For host-owned Dash applications:

```powershell
poetry install
poetry run python demos/03_viewer/03_embedded.py
```

This single file contains the app builder at the top and both geometry definitions.
Set `GEOMETRY = "curve"` or `"surface"` and `EDITABLE = True` or `False` to choose
any of the four combinations. `PORT` defaults to 8061. The builder selects
`GeometryEditor` or read-only `GeometryView`; both use the desktop renderer.
No application-analysis readout is added. The host can subscribe to accepted
model state for its own analysis.

Display opens projections, colours, diagnostics, grids, frames, and mesh controls.

For saved views and PNG output, run `04_figure_save.py`; it saves its files in
`output/04_figure_save/` beside the script. For a scripted sequence
showing an oloid's generating line and smooth cubic ruled patches, run
[`../oloid/demo_oloid_animation.py`](../oloid/demo_oloid_animation.py). It opens the
visualizer, then exports synchronized overview, line-following and side-by-side
PNG/GIF sequences when closed, with one-second pauses at each extreme.
Both use `nurbspy.Figure`; see the
[figure guide](../../docs/source/figures.md).
Edit opens direct X/Y/Z coordinate fields, constrained dragging, and weights.
Normal direction is fixed during a drag and refreshed afterward; it is available
in 3D and may be undefined at singular/straight portions. The status line explains
unavailable normals. XY/YZ/ZX mouse editing preserves the hidden coordinate.

Choose STL export mesh to generate the export and enable Download STL. Processing
and writing use main's external Trimesh exporter. Sampling is adjustable; an open
patch is not automatically converted to a printable solid.

The [torus/oloid printing example](../02_surfaces/15_stl_printing.py) remains
the separate multi-patch export example. Legacy Plotly demos were not ported.
See [the visualization guide](../../docs/source/visualization.md) and
[full behaviour specification](../../docs/VIEWER_EDITOR_BEHAVIOR.md).
