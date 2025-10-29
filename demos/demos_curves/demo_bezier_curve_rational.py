
""" Example showing how to create a rational Bezier curve """


# -------------------------------------------------------------------------------------------------------------------- #
# Importing packages
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import nurbspy.jax as nrb
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------------------------------------------- #
# 2D rational Bezier curve example
# -------------------------------------------------------------------------------------------------------------------- #
# Define the array of control points
P = np.zeros((2,5))
P[:, 0] = [0.20, 0.50]
P[:, 1] = [0.40, 0.70]
P[:, 2] = [0.80, 0.60]
P[:, 3] = [0.60, 0.20]
P[:, 4] = [0.40, 0.20]

# Define the array of control point weights
W = np.asarray([1, 1, 3, 1, 1])

# Create and plot the Bezier curve
bezier2D = nrb.NurbsCurve(control_points=P, weights=W)
bezier2D.plot(frenet_serret=True)


# -------------------------------------------------------------------------------------------------------------------- #
# 3D rational Bezier curve example
# -------------------------------------------------------------------------------------------------------------------- #
# Define the array of control points
P = np.zeros((3,5))
P[:, 0] = [0.00, 0.00, 0.00]
P[:, 1] = [0.00, 0.30, 0.05]
P[:, 2] = [0.25, 0.30, 0.30]
P[:, 3] = [0.50, 0.30, -0.05]
P[:, 4] = [0.50, 0.10, 0.10]

# Define the array of control point weights
W = np.asarray([1, 1, 3, 1, 1])

# Create and plot the rational Bezier curve
bezier3D = nrb.NurbsCurve(control_points=P, weights=W)
bezier3D.plot(frenet_serret=True)
plt.show()



