"""Shared Dash application builder for the separate embedded geometry demos."""

import numpy as np
from dash import Dash, html
from nurbspy.visualization import GeometryView, GeometryEditor


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
        resolution=45,
        height="min(74vh,780px)",
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
