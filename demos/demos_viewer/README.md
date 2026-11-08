# Shared viewer and editor examples

Run from the main checkout. Geometry uses Point inputs; arrays remain supported.
All settings are readable values in the scripts, with no command-line options.
The default appearance is solid colour; diagnostics are available in Display.

```powershell
poetry install
poetry run python demos/demos_viewer/demo_curve_view.py
poetry run python demos/demos_viewer/demo_curve_edit.py
poetry run python demos/demos_viewer/demo_surface_view.py
poetry run python demos/demos_viewer/demo_surface_edit.py
```

| Geometry | View | Edit |
| --- | --- | --- |
| Curve | [demo_curve_view.py](demo_curve_view.py) | [demo_curve_edit.py](demo_curve_edit.py) |
| Surface | [demo_surface_view.py](demo_surface_view.py) | [demo_surface_edit.py](demo_surface_edit.py) |

Desktop uses a native webview with no HTTP server. Apply returns a new NumPy
geometry, Cancel returns None, and original input remains unchanged. Save the
returned object with `to_yaml()` if desired. Load with the appropriate NumPy/JAX
class's `from_yaml()` before viewing. Undo/redo and file dialogs are omitted.

For host-owned Dash applications:

```powershell
poetry install
poetry run python demos/demos_viewer/demo_embedded_curve.py
poetry run python demos/demos_viewer/demo_embedded_surface.py
poetry run python demos/demos_viewer/demo_embedded_surface_view.py
```

These use ports 8061, 8062, and 8063 respectively. `_embedded_app.py` contains
support code, not another demo launcher. It selects `GeometryEditor` or read-only
`GeometryView`; both use the desktop renderer. No application-analysis readout
is added. The host can subscribe to accepted model state for its own analysis.

Display opens projections, colours, diagnostics, grids, frames, and mesh controls.

For saved views and PNG output, run `demo_figure_save.py`. For a scripted sequence
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

The [torus/oloid printing example](../demos_surfaces/demo_stl_printing.py) remains
the separate multi-patch export example. Legacy Plotly demos were not ported.
See [the visualization guide](../../docs/source/visualization.md) and
[full behaviour specification](../../docs/VIEWER_EDITOR_BEHAVIOR.md).
