
""" Example showing how to create a NURBS surface """


# -------------------------------------------------------------------------------------------------------------------- #
# Importing packages
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import nurbspy as nrb
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------------------------------------------- #
# NURBS surface example
# -------------------------------------------------------------------------------------------------------------------- #
# Define the control points
n_dim, n, m = 3, 5, 4
P = [
    [nrb.Point(0.0, 3.0, 0.0), nrb.Point(0.0, 3.0, 1.0), nrb.Point(0.0, 3.0, 2.0), nrb.Point(0.5, 3.0, 3.0)],
    [nrb.Point(1.0, 2.0, 0.0), nrb.Point(1.0, 2.0, 1.0), nrb.Point(1.0, 2.0, 2.0), nrb.Point(1.0, 2.5, 3.0)],
    [nrb.Point(2.0, 1.5, 0.0), nrb.Point(2.0, 1.5, 1.0), nrb.Point(2.0, 1.5, 2.0), nrb.Point(2.0, 2.0, 3.0)],
    [nrb.Point(3.0, 2.0, 0.0), nrb.Point(3.0, 2.0, 1.0), nrb.Point(3.0, 2.0, 2.0), nrb.Point(3.0, 2.5, 3.0)],
    [nrb.Point(4.0, 3.0, 0.0), nrb.Point(4.0, 3.0, 1.0), nrb.Point(4.0, 3.0, 2.0), nrb.Point(3.5, 3.0, 3.0)],
]  # points[u_index][v_index]

# Create and the NURBS surface
nurbsSurface = nrb.NurbsSurface(control_points=P)

# Define point to be projected
P0 = np.asarray([0.50, 0.50, 0.50]).reshape((3,1))

# Compute projected time
u0, v0 = nurbsSurface.project_point_to_surface(P0)
S0 = nurbsSurface.get_value(u0, v0)

# Plot the NURBS surface and the projected point
fig, ax = nurbsSurface.plot()
Px, Py, Pz = P0.flatten()
Sx, Sy, Sz = S0.flatten()
ax.plot([Px, Sx], [Py, Sy], [Pz, Sz], color='black', linestyle='--', marker='o', markeredgecolor='r', markerfacecolor='w')
plt.show()
