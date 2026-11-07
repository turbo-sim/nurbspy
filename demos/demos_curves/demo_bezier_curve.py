
""" Example showing how to create a Bezier curve """


# -------------------------------------------------------------------------------------------------------------------- #
# Importing packages
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import nurbspy.jax as nrb
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------------------------------------------- #
# 2D Bezier curve example
# -------------------------------------------------------------------------------------------------------------------- #
# Define the control points
P = [
    nrb.Point(0.2, 0.5),
    nrb.Point(0.4, 0.7),
    nrb.Point(0.8, 0.6),
    nrb.Point(0.6, 0.2),
    nrb.Point(0.4, 0.2),
]

# Create and plot the Bezier curve
bezier2D = nrb.NurbsCurve(control_points=P)
bezier2D.plot(frenet_serret=False)


# -------------------------------------------------------------------------------------------------------------------- #
# 3D Bezier curve example
# -------------------------------------------------------------------------------------------------------------------- #
# Define the control points
P = [
    nrb.Point(0.0, 0.0, 0.0),
    nrb.Point(0.0, 0.3, 0.05),
    nrb.Point(0.25, 0.3, 0.3),
    nrb.Point(0.5, 0.3, -0.05),
    nrb.Point(0.5, 0.1, 0.1),
]

# Create and plot the Bezier curve
bezier3D = nrb.NurbsCurve(control_points=P)
bezier3D.plot(frenet_serret=True)
plt.show()
