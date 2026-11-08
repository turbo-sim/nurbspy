# Reproducible figures and PNG frames

`nurbspy.Figure` stores named geometry snapshots, one shared camera, independent
object appearance and diagnostics, and a fixed figure aspect ratio. Desktop,
embedded Dash, and PNG output use the same Three.js renderer and Python geometry
evaluation. Matplotlib `.plot()` retains its existing interface.

## Save and restore an interactive view

```python
import nurbspy as nrb

curve = nrb.NurbsCurve([nrb.Point(-1, 0, 0), nrb.Point(0, 2, 1), nrb.Point(1, 0, 0)])
fig = nrb.Figure(curve, name="curve", width=1000, height=750)
fig.show()  # Adjust camera, projection, colours and diagnostics, then close.
fig.to_yaml("curve_view.yaml")

restored = nrb.Figure.from_yaml("curve_view.yaml")
restored.savefig("curve.png", width=2400, height=1800)
```

The YAML document contains geometry, not references to external files. It uses
the existing versioned geometry serialization inside a separate, versioned figure
document. `to_state()` and `Figure.from_state()` exchange equivalent independent
JSON-compatible dictionaries. YAML text streams are supported too.

`show()` captures camera and display changes on Close, Escape, or title-bar close.
`edit()` commits geometry and figure settings on Apply. Cancel or title-bar close
leaves the figure unchanged. Input geometry is copied, including JAX inputs;
accepted geometry snapshots use the NumPy backend. For a single object, `edit()`
returns the accepted geometry or `None`; for multiple objects it returns a name
mapping or `None`.

The convenience calls `geometry.show()` and `geometry.edit()` use temporary
figures. Retain an explicit `Figure` when settings need to survive the window.

## Several objects and independent appearance

```python
fig.add(other_curve, name="reference", display={
    "geometry_color": "#ff7f0e", "control_points": False, "control_polygon": False,
})
fig.update("reference", visible=False)
fig.update("curve", display={"quantity": "curvature", "colormap": "plasma",
                             "reverse": True, "limits": [0.0, 2.0]})
```

Object names are unique. The Object selector changes which object's settings and
editing controls are displayed. It preserves camera framing. Each object has its
own visibility, colours, control net/points, boundaries, diagnostics, frames and
mesh settings. One camera and one GPU renderer draw all visible objects.

For fitting comparisons, use `display={"opacity": 0.25}` for a translucent
reference and add read-only measurement points:

```python
fig.scatter(points, color="#263238", size=3.0, depth_test=False)
```

`points` has shape `(dimension, count)` or is a sequence of `Point` objects.
Two-dimensional coordinates use z=0. Marker size is in logical pixels;
`depth_test=False` keeps every measurement visible over the surfaces. Set it to
`True` for normal surface occlusion. These annotations are preserved in Figure
YAML and PNGs and do not appear among editable geometry objects.

`names` lists the objects; `geometry(name)` and `settings(name)` return independent
copies. Modify a geometry copy and pass it back through `update(name, geometry=...)`.
`remove(name)` removes an object. Omitting a name in `geometry()`, `settings()` or
`update()` uses the active object.

## Camera and output framing

```python
fig.set_camera(position=[4, -6, 3], target=[0, 0, 0], up=[0, 0, 1], zoom=1.2)
fig.rendering["background"] = "#ffffff"
```

The camera records position, target, up direction, perspective/orthographic
projection, zoom, field of view, clipping planes, orthographic half-height, and
navigation mode. Position/target/up together define orientation, including roll.
Projection choices are `3d`, `xy`, `yz`, and `zx`. Call `fit()` to request a fresh
fit to all visible objects on the next render. Otherwise restored or explicitly
set cameras are retained when geometry changes.

The figure's `width` and `height` define its logical size and aspect ratio.
`savefig()` accepts output pixel dimensions. If just one dimension is supplied,
the other follows the saved aspect ratio. A different output aspect ratio adds
background margins without changing the camera framing. Rendered grid, axes,
colour bars, and frame legends are included; inspector panes, header buttons,
and ViewCube are omitted. PNG is the supported output format; a binary stream
can be supplied instead of a path.

## Headless export and scripted animation

Playwright is a standard dependency. PNG export uses an isolated headless browser,
without opening a visible window or using your browser profile. It tries the
Playwright Chromium engine, then installed Microsoft Edge or Google Chrome. If
none is available, install the engine once:

```powershell
poetry run python -m playwright install chromium
```

No CDN or application server is needed. Export waits for full-quality geometry,
diagnostics, and fonts before capture. Exact pixels may vary across browser,
font, GPU, or library versions; the YAML preserves the scene parameters rather
than promising identical pixels across different platforms.

An ordinary Python loop can generate numbered frames:

```python
from pathlib import Path
import numpy as np

output = Path("frames")
output.mkdir(exist_ok=True)
fig.set_camera(position=[4, -6, 3], target=[0, 0, 0])
initial = fig.geometry("curve")
for index, displacement in enumerate(np.linspace(0, 1, 30)):
    moved = fig.geometry("curve")
    moved.P[:] = initial.P
    moved.P[2, 1] += displacement
    fig.update("curve", geometry=moved)
    fig.savefig(output / f"frame_{index:04d}.png", width=1600, height=1200)
```

Camera fields can also be changed per frame through `set_camera()`. Each
`savefig()` starts and closes an isolated browser. For a longer sequence, use
`fig.saveframes(figures, directory, width=..., height=..., progress=callback)` to
reuse one renderer. A generator can update and yield the same Figure for each
frame; each frame is saved before the generator advances. The optional callback
receives the one-based frame index and saved path. No animation class is required.

Run `demos/demos_viewer/demo_figure_save.py` for interactive capture, restoration, and PNG export.

## Embedded Dash figures

```python
from dash import Dash
from nurbspy.visualization import GeometryEditor

component = GeometryEditor(fig, id="geometry", height="75vh")
app = Dash(__name__)
app.layout = component.layout
component.register(app)
```

Use `GeometryView(fig, ...)` for read-only viewing. Subscribe to `component.state_id`
to receive the full figure snapshot, and call `component.figure_from_state(state)`
to reconstruct it for saving or export in a host callback. State belongs to each
browser's Store; interactions do not mutate the shared server-side input figure.
