# Reproducible figures and PNG frames

`nurbspy.Figure` stores named geometry snapshots, one shared camera, independent
object appearance and diagnostics, and a fixed figure aspect ratio. Desktop,
embedded Dash, and PNG output use the same Three.js renderer and Python geometry
evaluation. Matplotlib `.plot()` retains its existing interface.

## Automatic display quality

`Figure`, geometry `show()`/`edit()`, and embedded viewers choose display sampling
automatically. No resolution value is needed:

```python
fig = nrb.Figure(surface)                 # Auto balances detail and speed.
fig = nrb.Figure(surface, quality="fine") # Optional initial preference.
fig.update(display={"quality": "fast"}) # Change an existing object's quality.
```

Display > View > Display quality offers **Auto**, **Fast**, and **Fine**. The
preference belongs to the active object and is retained in Figure/YAML state.
It changes only the display approximation, leaving the NURBS geometry and STL
export settings unchanged.

The sampler includes knot boundaries within its sampling budget and probes chord
deviation and changes of direction. Surface sampling can differ along $u$ and
$v$; a twisted patch is also checked even when its parameter curves are straight.
Refinement is bounded to five passes and at most 128 samples per direction
(16,384 surface vertices). These are visual quality heuristics, not certified
geometric error bounds. Very complex geometry can reach the sampling limit.

Point and weight dragging reuse a coarse grid, capped at 20 samples per direction.
Only a completed edit reassesses the object's full sampling. Camera movement
reuses the existing GPU mesh and performs no Python geometry evaluation. A bounded
cache reuses sampling for recent edits and quality changes; shared hosts keep
accepted geometry separate for each browser.

For small curves where a coarse outline is distracting, keep the full display
grid during dragging:

```python
fig = nrb.Figure(curve, quality="fine", preview_quality="full")
edited = curve.edit(quality="fine", preview_quality="full")
```

`preview_quality="coarse"` remains the default. `"full"` reuses the last full
sampling grid without thinning it; a completed edit can still refine that grid.
The same option is available on `show()`, `Figure.show()`/`edit()`, `GeometryView`,
and `GeometryEditor`, and is retained in Figure/YAML state. It also works with
explicit numeric `resolution`. Use `preview_resolution` only with coarse previews;
combining it with `preview_quality="full"` raises an error. These options change
display approximation only, leaving control points and exported geometry intact.

Standalone HTML includes precomputed quality choices, so its Auto/Fast/Fine
selector works offline. This increases export preparation and file size, without
requiring a Python server while viewing. PNG output uses the selected quality.

An explicit `resolution=...` remains an advanced override for older scripts and
saved views. It retains the previous fixed-grid behavior and disables the quality
selector. Existing numeric-resolution YAML files continue to load.

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

Enable **Isocurves** under Display > Geometry for lines that follow the edited
surface. The parameter positions and colour belong to that surface's saved
appearance, so updates, undo/redo, YAML, PNG, and HTML use the same settings:

```python
fig = nrb.Figure(surface, display={"isocurves": True})
# Optional fixed native parameter values; an empty list disables one direction.
fig.update(display={"isocurves_u": [0.2, 0.5, 0.8], "isocurves_v": [],
                    "isocurves_color": "#536d7d"})
```

By default, five interior lines in each direction span 8% to 92% of the surface's
parameter domain. A $u$-isocurve fixes $u$ and varies $v$, matching
`surface.get_isocurve_u(u0)`. Explicit lists accept up to 30 increasing parameters
per direction within that direction's native knot domain. Isocurves use the
surface's current preview/full sampling axes and update after point and weight
edits. They are display overlays and have no independent editable control points.
They are drawn over the surface to avoid depth flicker. Standalone HTML retains
precomputed lines with working visibility, colour, and quality choices.

For independent curve annotations lying on a surface, set
`display={"curve_depth_test": False}` to draw them clearly over the surface.
This is the curve's **Hide behind surfaces** checkbox in Display. Normal occlusion
remains enabled by default. Adding a curve with `fig.add(surface.get_isocurve_u(...))`
still creates an independent snapshot; use the surface display option for live lines.

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

`fig.mesh(vertices, triangles, color="#b9cbd6", wireframe=False)` similarly adds a
read-only triangle mesh, such as an STL reloaded from disk. Vertices have shape
`(count, 3)` and triangles contain integer vertex-index triples. The mesh is
preserved in YAML, HTML, and PNG; `wireframe=True` also draws its triangle edges.

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

For a self-contained interactive page, use `fig.to_html("surface.html")`.
The file bundles the Three.js renderer and precomputed geometry; recipients
need a browser, with no Python server or internet access. Optional named
views use `fig.to_html("comparison.html", variants={"Initial": fig, "Final": other_fig})`.
Camera and display choices persist when switching views. Measurement markers
can be hidden independently. Geometry is read-only, and controls requiring
new Python diagnostics are disabled; exported diagnostics remain available.

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

Run `04_figure_save.py` for interactive capture/restoration or
`demos/oloid/demo_oloid_animation.py` for two reference circles, a moving straight
generating line, and progressively revealed cubic ruled surface patches.
It opens the visualizer before export and creates a seamless PNG/GIF sequence
with two cosine development/undevelopment cycles and one-second holds at the
fully formed and line-only states. It exports an overview GIF with three
continuous camera orbits, a line-following GIF, and a synchronized side-by-side
GIF. The following camera stays on the outward side of the active ruling, keeping
the construction visible. The GIF join is at the line-only state. Export
switches and all settings are global variables.
The oloid's cubic boundaries approximate the analytical circles with an adaptive,
shared-knot fit; the surface stays linear along its straight rulings.

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
