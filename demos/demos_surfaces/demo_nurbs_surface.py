
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
# Define the control points
n_dim, n, m = 3, 5, 4
P = [
    [nrb.Point(0.0, 3.0, 0.0), nrb.Point(0.0, 3.0, 1.0), nrb.Point(0.0, 3.0, 2.0), nrb.Point(0.5, 3.0, 3.0)],
    [nrb.Point(1.0, 2.0, 0.0), nrb.Point(1.0, 2.0, 1.0), nrb.Point(1.0, 2.0, 2.0), nrb.Point(1.0, 2.5, 3.0)],
    [nrb.Point(2.0, 1.5, 0.0), nrb.Point(2.0, 1.5, 1.0), nrb.Point(2.0, 1.5, 2.0), nrb.Point(2.0, 2.0, 3.0)],
    [nrb.Point(3.0, 2.0, 0.0), nrb.Point(3.0, 2.0, 1.0), nrb.Point(3.0, 2.0, 2.0), nrb.Point(3.0, 2.5, 3.0)],
    [nrb.Point(4.0, 3.0, 0.0), nrb.Point(4.0, 3.0, 1.0), nrb.Point(4.0, 3.0, 2.0), nrb.Point(3.5, 3.0, 3.0)],
]  # points[u_index][v_index]

# Define the array of control point weights
W = np.zeros((n, m))
W[:, 0] = np.asarray([1, 1, 5, 1, 1])
W[:, 1] = np.asarray([1, 1, 10, 1, 1])
W[:, 2] = np.asarray([1, 1, 10, 1, 1])
W[:, 3] = np.asarray([1, 1, 5, 1, 1])

# Maximum index of the control points (counting from zero)
n = len(P) - 1
m = len(P[0]) - 1

# Define the order of the basis polynomials
# Linear (p = 1), Quadratic (p = 2), Cubic (p = 3), etc.
# Set p = n (number of control points minus one) to obtain a Bezier
p = 3
q = 3

# Define the knot vectors (clamped spline)
# p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones.  In total r+1 points where r=n+p+1
# q+1 zeros, m-p equispaced points between 0 and 1, and q+1 ones. In total s+1 points where s=m+q+1
U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))
V = np.concatenate((np.zeros(q), np.linspace(0, 1, m - q + 2), np.ones(q)))

# Create and plot the NURBS surface
nurbsSurface = nrb.NurbsSurface(control_points=P, weights=W, u_degree=p, v_degree=q, u_knots=U, v_knots=V)
nurbsSurface.plot(control_points=True)
plt.show()
