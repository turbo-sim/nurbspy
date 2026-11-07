
""" Example showing how to create a B-Spline curve """


# -------------------------------------------------------------------------------------------------------------------- #
# Importing packages
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import nurbspy.jax as nrb
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------------------------------------------- #
# 2D B-Spline curve example
# -------------------------------------------------------------------------------------------------------------------- #
# Define the control points
P = [
    nrb.Point(0.2, 0.5),
    nrb.Point(0.4, 0.7),
    nrb.Point(0.8, 0.6),
    nrb.Point(0.6, 0.2),
    nrb.Point(0.4, 0.2),
]

# Maximum index of the control points (counting from zero)
n = len(P) - 1

# Define the order of the basis polynomials
p = 2

# Define the knot vector (clamped spline)
# p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones. In total r+1 points where r=n+p+1
U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

# Create and plot the B-Spline curve
bspline2D = nrb.NurbsCurve(control_points=P, degree=p, knots=U)
bspline2D.plot(frenet_serret=True)


# -------------------------------------------------------------------------------------------------------------------- #
# 3D B-Spline curve example
# -------------------------------------------------------------------------------------------------------------------- #
# Define the control points
P = [
    nrb.Point(0.0, 0.0, 0.0),
    nrb.Point(0.0, 0.3, 0.05),
    nrb.Point(0.25, 0.3, 0.3),
    nrb.Point(0.5, 0.3, -0.05),
    nrb.Point(0.5, 0.1, 0.1),
]

# Define the order of the basis polynomials
p = 2

# Create and plot the B-Spline curve (note that it is possible to create the curve not specifying the knot vector)
bspline3D = nrb.NurbsCurve(control_points=P, degree=p)
bspline3D.plot(frenet_serret=True)
plt.show()
