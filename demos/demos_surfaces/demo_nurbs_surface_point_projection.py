
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
# Define the array of control points
n_dim, n, m = 3, 5, 4
P = np.zeros((n_dim, n, m))

# First row
P[:, 0, 0] = [0.00, 3.00, 0.00]
P[:, 1, 0] = [1.00, 2.00, 0.00]
P[:, 2, 0] = [2.00, 1.50, 0.00]
P[:, 3, 0] = [3.00, 2.00, 0.00]
P[:, 4, 0] = [4.00, 3.00, 0.00]

# Second row
P[:, 0, 1] = [0.00, 3.00, 1.00]
P[:, 1, 1] = [1.00, 2.00, 1.00]
P[:, 2, 1] = [2.00, 1.50, 1.00]
P[:, 3, 1] = [3.00, 2.00, 1.00]
P[:, 4, 1] = [4.00, 3.00, 1.00]

# Third row
P[:, 0, 2] = [0.00, 3.00, 2.00]
P[:, 1, 2] = [1.00, 2.00, 2.00]
P[:, 2, 2] = [2.00, 1.50, 2.00]
P[:, 3, 2] = [3.00, 2.00, 2.00]
P[:, 4, 2] = [4.00, 3.00, 2.00]

# Fourth row
P[:, 0, 3] = [0.50, 3.00, 3.00]
P[:, 1, 3] = [1.00, 2.50, 3.00]
P[:, 2, 3] = [2.00, 2.00, 3.00]
P[:, 3, 3] = [3.00, 2.50, 3.00]
P[:, 4, 3] = [3.50, 3.00, 3.00]

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


