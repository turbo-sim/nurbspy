"""View a NURBS surface; all options are defined in this script."""

import numpy as np
import nurbspy as nrb

# A weighted bicubic saddle. Set z to zero to try a planar surface.
x, y = np.meshgrid(np.linspace(-1.5, 1.5, 4), np.linspace(-1.5, 1.5, 4), indexing="ij")
z = 0.45 * (x * x - y * y)
weights = np.ones((4, 4))
weights[1:3, 1:3] = 1.8
points = [[nrb.Point(x[i, j], y[i, j], z[i, j]) for j in range(4)] for i in range(4)]
surface = nrb.NurbsSurface(points, weights=weights)

# Enable a mesh in Display, or set mesh="display" or mesh="export" explicitly.
display = dict(
    view="3d",
    quantity="solid",
    colormap="coolwarm",
    mesh="off",
    mesh_u=40,
    mesh_v=40,
)

surface.show(title="NURBS surface viewer", resolution=45, display=display)
