
""" Example showing how to combine several NURBS curves into a single NURBS curve
    The NURBS curve must be of the same order to merge them
    Merging curves of different degree would require degree elevation (not implemented yet)

"""


# -------------------------------------------------------------------------------------------------------------------- #
# Importing packages
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import nurbspy.jax as nrb
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------------------------------------------- #
# Create the first NURBS curve
# -------------------------------------------------------------------------------------------------------------------- #
# Define the control points
P = [
    nrb.Point(0.0, 0.0),
    nrb.Point(0.1, 0.5),
    nrb.Point(0.2, -0.5),
    nrb.Point(0.3, 0.0),
    nrb.Point(0.4, 0.1),
    nrb.Point(0.5, 0.2),
    nrb.Point(0.6, 0.3),
    nrb.Point(0.75, 0.0),
]

# Define the array of control weights
W = np.asarray([1, 1, 1, 1, 2, 2, 2, 2])

# Highest index of the control points (counting from zero)
n = len(P) - 1

# Define the order of the basis polynomials
p = 3

# Define the knot vector (clamped spline)
# p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones.  In total r+1 points where r=n+p+1
U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

# Create the B-Spline curve
nurbs1 = nrb.NurbsCurve(control_points=P, weights=W, degree=p, knots=U)


# -------------------------------------------------------------------------------------------------------------------- #
# Create the second NURBS curve
# -------------------------------------------------------------------------------------------------------------------- #
# Define the control points
P = [
    nrb.Point(0.75, 0.0),
    nrb.Point(1.0, 0.5),
    nrb.Point(1.25, -0.5),
    nrb.Point(1.5, 0.0),
]

# Define the array of control weights
W = np.asarray([5, 2, 2, 4])

# Highest index of the control points (counting from zero)
n = len(P) - 1

# Define the order of the basis polynomials
p = 3

# Define the knot vector (clamped spline)
# p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones.  In total r+1 points where r=n+p+1
U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

# Create the B-Spline curve
nurbs2 = nrb.NurbsCurve(control_points=P, weights=W, degree=p, knots=U)


# -------------------------------------------------------------------------------------------------------------------- #
# Create the third NURBS curve
# -------------------------------------------------------------------------------------------------------------------- #
# Define the control points
P = [
    nrb.Point(1.5, 0.0),
    nrb.Point(2.0, 0.5),
    nrb.Point(1.5, 0.75),
    nrb.Point(1.0, 0.6),
    nrb.Point(0.5, 0.3),
]

# Define the array of control weights
W = np.asarray([1, 1, 1, 1, 1])

# Highest index of the control points (counting from zero)
n = len(P) - 1

# Define the order of the basis polynomials
p = 3

# Define the knot vector (clamped spline)
# p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones.  In total r+1 points where r=n+p+1
U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

# Create the B-Spline curve
nurbs3 = nrb.NurbsCurve(control_points=P, weights=W, degree=p, knots=U)


# -------------------------------------------------------------------------------------------------------------------- #
# Merge the NURBS curves
# -------------------------------------------------------------------------------------------------------------------- #
nurbsMerged = nrb.merge_nurbs_curves([nurbs1, nurbs2, nurbs3])


# -------------------------------------------------------------------------------------------------------------------- #
# Plot the NURBS curves
# -------------------------------------------------------------------------------------------------------------------- #
fig, ax = nurbsMerged.plot(curve=False, control_points=False)
nurbs1.plot_curve(fig, ax, color='green', linewidth=2)
nurbs2.plot_curve(fig, ax, color='red',   linewidth=2)
nurbs3.plot_curve(fig, ax, color='blue',  linewidth=2)
nurbsMerged.plot_curve(fig, ax, color='w', linewidth=1, linestyle=':')
plt.show()
