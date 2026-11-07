
""" Example showing how to create a Coons NURBS surface """


# -------------------------------------------------------------------------------------------------------------------- #
# Importing packages
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import nurbspy as nrb
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------------------------------------------- #
# Coons surface example
# -------------------------------------------------------------------------------------------------------------------- #

# Define the south boundary (rational Bézier curve)
P = [
    nrb.Point(0.0, 0.0, 0.0),
    nrb.Point(0.33, 0.0, -0.4),
    nrb.Point(0.66, 0.1, 0.6),
    nrb.Point(1.0, 0.2, 0.4),
]
W = np.asarray([1, 2, 2, 1])
nurbsCurve_south = nrb.NurbsCurve(control_points=P, weights=W)

# Define the north boundary (rational Bézier curve)
P = [
    nrb.Point(0.05, 1.0, 0.0),
    nrb.Point(0.33, 1.15, 0.4),
    nrb.Point(0.66, 1.15, 0.0),
    nrb.Point(1.05, 1.25, 0.4),
]
W = np.asarray([1, 2, 2, 1])
nurbsCurve_north = nrb.NurbsCurve(control_points=P, weights=W)


# Define the west boundary (rational Bézier curve)
P = [
    nrb.Point(*nurbsCurve_south.P[:, 0]),
    nrb.Point(-0.2, 0.5, -0.4),
    nrb.Point(*nurbsCurve_north.P[:, 0]),
]
W = np.asarray([nurbsCurve_south.W[0], 1, nurbsCurve_north.W[0]])
nurbsCurve_west = nrb.NurbsCurve(control_points=P, weights=W)


# Define the east boundary (rational Bézier curve)
P = [
    nrb.Point(*nurbsCurve_south.P[:, -1]),
    nrb.Point(1.15, 0.5, 0.3),
    nrb.Point(*nurbsCurve_north.P[:, -1]),
]
W = np.asarray([nurbsCurve_south.W[-1], 1, nurbsCurve_north.W[-1]])
nurbsCurve_east = nrb.NurbsCurve(control_points=P, weights=W)


# Create and plot the Coons NURBS surface
coonsNurbsSurface = nrb.NurbsSurfaceCoons(nurbsCurve_south, nurbsCurve_north, nurbsCurve_west, nurbsCurve_east).NurbsSurface
fig, ax = coonsNurbsSurface.plot(surface=True, surface_color='blue', control_points=False)


# Plot isoparametric curves
coonsNurbsSurface.plot_isocurve_u(fig, ax, np.linspace(0, 1, 5))
coonsNurbsSurface.plot_isocurve_v(fig, ax, np.linspace(0, 1, 5))


# Plot the boundary NURBS curves
nurbsCurve_south.plot_curve(fig, ax, color='b', linewidth=2.5)
nurbsCurve_north.plot_curve(fig, ax, color='b', linewidth=2.5)
nurbsCurve_west.plot_curve(fig, ax, color='b', linewidth=2.5)
nurbsCurve_east.plot_curve(fig, ax, color='b', linewidth=2.5)
plt.show()
