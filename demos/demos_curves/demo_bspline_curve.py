
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
# Define the array of control points
P = np.zeros((2,5))
P[:, 0] = [0.20, 0.50]
P[:, 1] = [0.40, 0.70]
P[:, 2] = [0.80, 0.60]
P[:, 3] = [0.60, 0.20]
P[:, 4] = [0.40, 0.20]

# Maximum index of the control points (counting from zero)
n = np.shape(P)[1] - 1

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
# Define the array of control points
P = np.zeros((3,5))
P[:, 0] = [0.00, 0.00, 0.00]
P[:, 1] = [0.00, 0.30, 0.05]
P[:, 2] = [0.25, 0.30, 0.30]
P[:, 3] = [0.50, 0.30, -0.05]
P[:, 4] = [0.50, 0.10, 0.10]

# Define the order of the basis polynomials
p = 2

# Create and plot the B-Spline curve (note that it is possible to create the curve not specifying the knot vector)
bspline3D = nrb.NurbsCurve(control_points=P, degree=p)
bspline3D.plot(frenet_serret=True)
plt.show()



