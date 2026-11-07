"""View a NURBS curve; all options are defined in this script."""

from pathlib import Path
import sys

# Run this checkout directly, even if another nurbspy is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import nurbspy as nrb

# A sequence of Points avoids array-orientation ambiguity.
points = [
    nrb.Point(-2.0, 0.0, 0.0),
    nrb.Point(-1.0, 1.8, 0.5),
    nrb.Point(0.0, -1.0, 1.2),
    nrb.Point(1.0, 1.5, 0.4),
    nrb.Point(2.0, 0.0, 0.0),
]
curve = nrb.NurbsCurve(points, weights=np.array([1, 1, 0.6, 1.8, 1]), degree=3)

# Change these settings here, or use the Display menu in the viewer.
display = dict(
    view="3d",
    quantity="solid",
    colormap="viridis",
    frames=False,
    frame_count=7,
    frame_scale=0.08,
)

curve.show(title="NURBS curve viewer", resolution=45, display=display)
