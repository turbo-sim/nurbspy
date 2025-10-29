
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

# Maximum index of the control points (counting from zero)
n = np.shape(P)[1] - 1

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
# Define the array of control points
P = np.zeros((3,5))
P[:, 0] = [0.00, 0.00, 0.00]
P[:, 1] = [0.00, 0.30, 0.05]
P[:, 2] = [0.25, 0.30, 0.30]
P[:, 3] = [0.50, 0.30, -0.05]
P[:, 4] = [0.50, 0.10, 0.10]

# Maximum index of the control points (counting from zero)
n = np.shape(P)[1] - 1

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


