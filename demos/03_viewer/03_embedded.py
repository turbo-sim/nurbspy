"""Embed a curve or surface in Dash, with viewing or editing selected below."""

import numpy as np
import nurbspy as nrb
from dash import Dash, html
from nurbspy.visualization import GeometryView, GeometryEditor

GEOMETRY = "surface"  # Which example to build: "curve" or "surface".
EDITABLE = True  # True allows point/weight editing; False only views the geometry.
PORT = 8061  # Open http://127.0.0.1:8061; choose another unused port if needed.
DISPLAY = dict(
    view="3d",  # Perspective view; "xy", "yz", "zx" look straight at those planes.
    # "solid" uses one colour. Alternatives: "curvature" for a curve,
    # or "mean_curvature" for a surface.
    quantity="solid",
    colormap="viridis",  # Colour palette for quantities; unused with "solid".
    mesh="off",  # Surface mesh overlay: "off", "display", or "export" (STL preview).
)


def build_app(geometry, *, title=None, display=None, editable=True):
    """Embed one shared viewer/editor without application-specific readouts."""
    if not isinstance(editable, bool):
        raise ValueError("editable must be a boolean")
    kind = "curve" if np.ndim(geometry.P) == 2 else "surface"
    action = "editing" if editable else "viewing"
    title = title or f"{kind.title()} {action} inside a research application"
    component = GeometryEditor if editable else GeometryView
    editor = component(
        geometry,
        id=kind,
        title=f"{kind.title()} {'editor' if editable else 'viewer'}",
        height="min(74vh,780px)",  # 74% of browser height, capped at 780 pixels.
        display=display,
    )
    app = Dash(__name__)
    app.layout = html.Main(
        [
            html.H1(title, style={"fontSize": "26px", "marginBottom": "10px"}),
            html.P(
                "Drag control points or use Edit to enter coordinates and weights."
                if editable
                else "Explore the geometry, projections, and scientific display options."
            ),
            editor.layout,
        ],
        style={
            "maxWidth": "1400px",
            "padding": "20px 24px",
            "margin": "auto",
            "fontFamily": "system-ui",
            "color": "#203440",
        },
    )

    editor.register(app)
    return app


def make_geometry(kind):
    """Define the two example geometries in one place, using Point inputs."""
    if kind == "curve":
        points = [
            nrb.Point(-2.0, 0.0, 0.0),
            nrb.Point(-1.0, 1.8, 0.5),
            nrb.Point(0.0, -1.0, 1.2),
            nrb.Point(1.0, 1.5, 0.4),
            nrb.Point(2.0, 0.0, 0.0),
        ]
        return nrb.NurbsCurve(points, weights=np.array([1, 1, 0.6, 1.8, 1]), degree=3)
    if kind == "surface":
        # A weighted bicubic saddle. Set z to zero to inspect a planar surface.
        x, y = np.meshgrid(
            np.linspace(-1.5, 1.5, 4), np.linspace(-1.5, 1.5, 4), indexing="ij"
        )
        z = 0.45 * (x * x - y * y)
        weights = np.ones((4, 4))
        weights[1:3, 1:3] = 1.8
        points = [
            [nrb.Point(x[i, j], y[i, j], z[i, j]) for j in range(4)] for i in range(4)
        ]
        return nrb.NurbsSurface(points, weights=weights)
    raise ValueError('GEOMETRY must be "curve" or "surface"')


def main():
    geometry = make_geometry(GEOMETRY)
    app = build_app(
        geometry,
        display=DISPLAY,
        editable=EDITABLE,
    )
    app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
