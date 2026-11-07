"""A standalone embedded surface example; settings are defined here."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
import nurbspy as nrb
from _embedded_app import build_app

# A spatial saddle. Set z to zero to inspect a planar surface.
x, y = np.meshgrid(np.linspace(-1.5, 1.5, 4), np.linspace(-1.5, 1.5, 4), indexing="ij")
z = 0.45 * (x * x - y * y)
weights = np.ones((4, 4))
weights[1:3, 1:3] = 1.8
points = [[nrb.Point(x[i, j], y[i, j], z[i, j]) for j in range(4)] for i in range(4)]
surface = nrb.NurbsSurface(points, weights=weights)
display = dict(view="3d", quantity="solid", colormap="viridis")
# Set editable=False to use the same component as a read-only viewer.
app = build_app(surface, display=display, editable=True)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8062, debug=False, use_reloader=False)
