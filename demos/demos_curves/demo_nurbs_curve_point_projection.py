
""" Example showing how to create a NURBS curve """


# -------------------------------------------------------------------------------------------------------------------- #
# Importing packages
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import nurbspy.jax as nrb
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------------------------------------------- #
# 2D NURBS curve example
# -------------------------------------------------------------------------------------------------------------------- #
# Define the array of control points
P = np.zeros((2,5))
P[:, 0] = [0.20, 0.50]
P[:, 1] = [0.40, 0.70]
P[:, 2] = [0.80, 0.60]
P[:, 3] = [0.60, 0.20]
P[:, 4] = [0.40, 0.20]

# Create the NURBS curve
nurbs2D = nrb.NurbsCurve(control_points=P, degree=3)

# Define point to be projected
P0 = np.asarray([0.50, 0.50])

# Compute projected time
u0 = nurbs2D.project_point_to_curve(P0)
C0 = nurbs2D.get_value(u0)

# Plot the NURBS curve and the projected point
fig, ax = nurbs2D.plot()
Px, Py = P0.flatten()
Cx, Cy = C0.flatten()
ax.plot([Px, Cx], [Py, Cy], color='black', linestyle='--', marker='o', markeredgecolor='b', markerfacecolor='w')
plt.show()

