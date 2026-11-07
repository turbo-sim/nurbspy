
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
# Define the control points
P = [
    nrb.Point(0.2, 0.5),
    nrb.Point(0.4, 0.7),
    nrb.Point(0.8, 0.6),
    nrb.Point(0.6, 0.2),
    nrb.Point(0.4, 0.2),
]

# Create the NURBS curve
nurbs2D = nrb.NurbsCurve(control_points=P, degree=3)

# Define point to be projected
P0 = np.asarray([0.50, 0.50])

# Compute projected time
u0 = nurbs2D._project_point_scalar(P0)
C0 = nurbs2D.get_value(u0)

# Plot the NURBS curve and the projected point
fig, ax = nurbs2D.plot()
Px, Py = P0.flatten()
Cx, Cy = C0.flatten()
ax.plot([Px, Cx], [Py, Cy], color='black', linestyle='--', marker='o', markeredgecolor='b', markerfacecolor='w')
plt.show()
