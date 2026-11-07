
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

# Maximum index of the control points (counting from zero)
n = len(P) - 1

# Define the array of control point weights
W = np.asarray([1, 1, 3, 1, 1])

# Define the order of the basis polynomials
# Linear (p = 1), Quadratic (p = 2), Cubic (p = 3), etc.
# Set p = n (number of control points minus one) to obtain a Bezier
p = 2

# Define the knot vector (clamped spline)
# p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones. In total r+1 points where r=n+p+1
U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

# Create and plot the NURBS curve
nurbs2D = nrb.NurbsCurve(control_points=P, weights=W, degree=p, knots=U)
nurbs2D.plot(frenet_serret=True)


# -------------------------------------------------------------------------------------------------------------------- #
# 3D NURBS curve example
# -------------------------------------------------------------------------------------------------------------------- #
# Define the control points
P = [
    nrb.Point(0.0, 0.0, 0.0),
    nrb.Point(0.0, 0.3, 0.05),
    nrb.Point(0.25, 0.3, 0.3),
    nrb.Point(0.5, 0.3, -0.05),
    nrb.Point(0.5, 0.1, 0.1),
]

# Maximum index of the control points (counting from zero)
n = len(P) - 1

# Define the array of control point weights
W = np.asarray([1, 1, 0.5, 1, 1])

# Define the order of the basis polynomials
# Linear (p = 1), Quadratic (p = 2), Cubic (p = 3), etc.
# Set p = n (number of control points minus one) to obtain a Bezier
p = 2

# Define the knot vector (clamped spline)
# p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones. In total r+1 points where r=n+p+1
U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

# Create the NURBS curve
nurbs3D = nrb.NurbsCurve(control_points=P, weights=W, degree=p, knots=U)
nurbs3D.plot(frenet_serret=True)
plt.show()
