# Interactive geometry viewing and editing

Curves and surfaces share one interactive renderer, with optional editing.
Numerical evaluation stays in Python; NumPy and JAX inputs use the same interface.
The implementation is in `nurbspy.visualization`, with thin geometry methods.
Main's Point inputs, YAML exchange, and external mesh exporter are preserved.

For persistent camera/display settings, multiple named geometries, PNG export,
and scripted animation frames, see [Reproducible figures](figures.md).

## Desktop

Desktop visualization and Dash embedding are standard dependencies. From the checkout:

```powershell
poetry install
```

For a release containing these changes, use `pip install nurbspy`.

```python
import nurbspy as nrb

curve = nrb.NurbsCurve([
    nrb.Point(0, 0),
    nrb.Point(1, 2),
    nrb.Point(3, 0),
])
curve.show()
edited = curve.edit()
if edited is not None:
    edited.to_yaml("edited_curve.yaml")
```

`show()` opens a read-only native window and returns `None` on close. `edit()`
returns a new NumPy geometry on Apply, or `None` on cancellation. Inputs remain
unchanged, including JAX objects. Two-coordinate inputs are promoted to spatial
geometry with zero Z at the visualization boundary. Accepted edits are outside
JIT/autodiff; YAML can restore them through either numerical backend.

The native window uses pywebview and a direct Python bridge. It starts no
Flask/Dash app, local HTTP asset server, or external browser. Call it on the main
thread; calls block until the window closes. Windows uses WebView2; other systems
need their supported native webview runtime. Native UI was previously checked
on Windows; macOS/Linux interactive behaviour is not claimed as verified here.

Equivalent function-oriented entry points are available:

```python
from nurbspy.visualization import show, edit, plot
show(curve, display=dict(view="xy", axes=True))
edited = edit(curve)
figure, axes = plot(curve)                 # Optional static Matplotlib figure
show(curve, backend="matplotlib")          # Explicit static window
```

Existing `.plot()` signatures and diagnostic plots remain available. The primary
interactive geometry examples use `.show()` and `.edit()`.

## Embedded in a research application

```powershell
poetry install
```

```python
from dash import Dash, html
from nurbspy.visualization import GeometryEditor

editor = GeometryEditor(curve, id="blade")
app = Dash(__name__)
app.layout = html.Main([editor.layout])
editor.register(app)

if __name__ == "__main__":
    app.run(debug=False)
```

Use `GeometryView` for read-only inspection. It omits editing controls and edit
endpoints while retaining display/diagnostic options. The host owns its app,
server, layout, authentication, and analyses; the component starts no server.
Register each unique ID once before serving requests. Application URL prefixes
and multiple independent components are supported.

Subscribe to `editor.state_id` in a Dash callback to consume full accepted models.
`editor.geometry_from_state(state)` reconstructs a native NumPy geometry. Only
committed full-quality updates publish state after the edit queue drains;
transient previews do not trigger expensive host analysis. The examples omit
application-specific readouts. Geometry save/load remains ordinary Python code.

## Controls and appearance

- Solid colour by default: black curves, blue surfaces, orange control net/points,
  and a white background. Editors highlight the active point in green and enlarge
  it slightly; viewers keep all points in their normal colour and size, including
  after clicking a point. Display has two colour pickers in viewers
  and an additional selected-point colour picker in editors.
- The horizontal header contains the Navigation dropdown, Fit, Reset in editors,
  Display, and Edit in editors. It wraps in narrow containers.
- Display/Edit open the right inspector; clicking the active button, its close
  button, or Escape closes it. Switching panes preserves model/camera/settings.
- Inspector inputs share aligned label/control columns and consistent field
  widths. In narrow panes, numeric fields and dropdowns stack below their labels.
- Navigation hints, coordinate footers, and desktop explanatory messages are
  omitted in both modes. Error feedback and action buttons remain available.
- Navigation offers Arcball, Orbit, and Trackball. The ViewCube supplies face,
  edge, corner, orbit/roll, Home, and isometric orientations.
- 3D, XY, YZ, and ZX are views of one spatial model. Planar views lock mouse
  motion to the selected plane and optionally show ticked axes.
- Grids render behind geometry. Orthographic points have white faces and coloured
  outlines; glossy spatial spheres match their apparent size through zoom/depth.
- Edit includes X coordinate, Y coordinate, Z coordinate, positive weight inputs,
  and axis/plane/Normal direction drag constraints. Press Enter or leave a field
  to commit. Reset restores the original coordinates/weights.

Normal direction is available in 3D. Python evaluates the curve Frenet normal or
surface normal at a Greville parameter associated with the point. The direction
is fixed for each drag and refreshed afterward. Undefined normals are reported
instead of guessed. Orthographic dragging preserves the hidden coordinate;
numeric fields can explicitly change any coordinate.

## Diagnostics and meshes

Display offers curve curvature, torsion, cumulative arc length, parameter speed,
and u; surfaces offer mean/Gaussian and principal curvatures, u, and v. Parameter
speed is $\|dC/du\|$, not physical velocity. Colormaps include viridis, plasma,
magma, inferno, cividis, coolwarm, and RdBu, with reversal, limits, and a colour bar.
Undefined values are grey. Curve Frenet frames are opt-in.

Mesh overlays are off by default. Select a mesh in Display to enable it.
Display mesh shows rendering triangles. STL export mesh generates a processed
export preview and enables Download STL. Both use main's Trimesh/NetworkX export
pipeline, including its validation. The old experimental writer is not required.
The current edited model is exported. Samples along u/v are configurable;
STL remains unitless and sampled. Open patches remain open. See
[geometry I/O](geometry_io.md) for exporter limits and code-level save/load.

Editing uses lower-detail previews and a full-quality update on release. Requests
coalesce; stale results are ignored. Invalid edits roll back to accepted geometry.
The GUI accepts real finite geometry and positive weights; numerical constructors
retain their broader complex/arbitrary-dimensional support.

Undo/redo, file dialogs, trimming, control-point insertion/deletion, and continuity
constraints across several patches are outside this integration's scope.

## Options and examples

Display quality defaults to **Auto**, with **Fast** and **Fine** alternatives in
Display. Geometry-aware sampling is cached during interaction and reassessed on
edit release. Numeric `resolution` remains an advanced fixed-sampling override.
See [automatic display quality](figures.md#automatic-display-quality).

The complete current behaviour is available as
{download}`the main behaviour specification <../VIEWER_EDITOR_BEHAVIOR.md>`. Runnable demos
are indexed in {download}`the demo index <../../demos/demos_viewer/README.md>`. Every current
script defines geometry/settings directly, without CLI arguments.
