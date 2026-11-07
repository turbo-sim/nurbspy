
""" Example showing how to create a B-Spline surface """


# -------------------------------------------------------------------------------------------------------------------- #
# Importing packages
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import nurbspy as nrb
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------------------------------------------- #
# B-Spline surface example
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

# Create and plot the B-Spline surface
bsplineSurface = nrb.NurbsSurface(control_points=P, u_degree=p, v_degree=q, u_knots=U, v_knots=V)
fig, ax = bsplineSurface.plot(surface=False, control_points=False)

# Plot the blade surface using a colormap based on curvature
bsplineSurface.plot_surface(fig, ax, color='mean_curvature', colorbar=True)

# Plot the vectors normal to the surface
bsplineSurface.plot_normals(fig, ax, scale=0.30)

# Plot isoparametric curves
bsplineSurface.plot_isocurve_u(fig, ax, np.linspace(0, 1, 5))
bsplineSurface.plot_isocurve_v(fig, ax, np.linspace(0, 1, 5))

# Show the figure
plt.show()
