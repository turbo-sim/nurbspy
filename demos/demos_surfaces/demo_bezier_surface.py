
""" Example showing how to create a polynomial Bezier surface """


# -------------------------------------------------------------------------------------------------------------------- #
# Importing packages
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import nurbspy as nrb
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------------------------------------------- #
# Polynomial Bezier surface example
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

# Create and plot the Bezier surface
bezierSurface = nrb.NurbsSurface(control_points=P)
bezierSurface.plot(control_points=True)
plt.show()
