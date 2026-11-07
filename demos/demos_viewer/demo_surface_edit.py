"""Edit a NURBS surface; all options are defined in this script."""

from pathlib import Path
import sys

# Run this checkout directly, even if another nurbspy is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import nurbspy as nrb

# A weighted bicubic saddle. Set z to zero to try a planar surface.
x, y = np.meshgrid(np.linspace(-1.5, 1.5, 4), np.linspace(-1.5, 1.5, 4), indexing="ij")
z = 0.45 * (x * x - y * y)
weights = np.ones((4, 4))
weights[1:3, 1:3] = 1.8
points = [[nrb.Point(x[i, j], y[i, j], z[i, j]) for j in range(4)] for i in range(4)]
surface = nrb.NurbsSurface(points, weights=weights)

# Choose mesh="export" to generate the STL mesh and enable Download STL.
display = dict(
    view="3d",
    quantity="solid",
    colormap="coolwarm",
    mesh="display",
    mesh_u=40,
    mesh_v=40,
)

edited = surface.edit(title="NURBS surface editor", resolution=45, display=display)
if edited is not None:
    print("Edited control points:\n", edited.P.T)
    print("Edited weights:\n", edited.W)

# Save from Python if desired:
# if edited is not None:
#     edited.to_yaml("edited_geometry.yaml")
