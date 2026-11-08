"""A standalone embedded curve example; settings are defined here."""

import numpy as np
import nurbspy as nrb
from _embedded_app import build_app

# Each Point defines one spatial control point.
points = [
    nrb.Point(-2.0, 0.0, 0.0),
    nrb.Point(-1.0, 1.8, 0.5),
    nrb.Point(0.0, -1.0, 1.2),
    nrb.Point(1.0, 1.5, 0.4),
    nrb.Point(2.0, 0.0, 0.0),
]
curve = nrb.NurbsCurve(points, weights=np.array([1, 1, 0.6, 1.8, 1]), degree=3)
display = dict(view="xy", axes=True, quantity="solid", colormap="viridis")
# Set editable=False to use the same component as a read-only viewer.
app = build_app(curve, display=display, editable=True)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8061, debug=False, use_reloader=False)
