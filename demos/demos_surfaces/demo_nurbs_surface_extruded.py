
""" Example showing how to create an extruded NURBS surface (general cylinder) """


# -------------------------------------------------------------------------------------------------------------------- #
# Importing packages
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import nurbspy as nrb
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------------------------------------------- #
# Extruded surface example:
# -------------------------------------------------------------------------------------------------------------------- #
# Define the base NURBS curve (rational Bézier curve)
P = [
    nrb.Point(0.0, 0.0, 0.0),
    nrb.Point(0.5, 0.8, 0.0),
    nrb.Point(0.75, 0.6, 0.0),
    nrb.Point(1.0, 0.3, 0.0),
    nrb.Point(0.8, 0.1, 0.0),
]
W = np.asarray([1, 1, 1, 1, 1])
nurbsCurve = nrb.NurbsCurve(control_points=P, weights=W)

# Set the extrusion direction (can be unitary or not)
direction = np.asarray([1, 1, 5])

# Set the extrusion length
length = 1

# Create and plot the ruled NURBS surface
extrudedNurbsSurface = nrb.NurbsSurfaceExtruded(nurbsCurve, direction, length).NurbsSurface
fig, ax = extrudedNurbsSurface.plot(surface=True, surface_color='blue', control_points=True)

# Plot isoparametric curves
extrudedNurbsSurface.plot_isocurve_u(fig, ax, np.linspace(0, 1, 5))
extrudedNurbsSurface.plot_isocurve_v(fig, ax, np.linspace(0, 1, 5))

# Plot the base NURBS curve
nurbsCurve.plot_curve(fig, ax, color='b', linewidth=2.5)



# -------------------------------------------------------------------------------------------------------------------- #
# Extruded surface example: A circular cylinder
# -------------------------------------------------------------------------------------------------------------------- #

# Define a circular arc
O = nrb.Point(0.0, 0.0, 0.0)
X = np.asarray([1.00, 0.00, 0.00])      # Abscissa direction
Y = np.asarray([0.00, 1.00, 0.00])      # Ordinate direction
R = 0.5                                 # Circle radius
theta_start = 0.00                      # Start angle
theta_end = 2*np.pi                     # End angle

# Create and plot the circular arc
nurbsCurve = nrb.CircularArc(O, X, Y , R, theta_start, theta_end).NurbsCurve

# Set the extrusion direction (can be unitary or not)
direction = np.asarray([0, 0, 1])

# Set the extrusion length
length = 1

# Create and plot the ruled NURBS surface
extrudedNurbsSurface = nrb.NurbsSurfaceExtruded(nurbsCurve, direction, length).NurbsSurface
fig, ax = extrudedNurbsSurface.plot(surface=True, surface_color='blue', control_points=True)

# Plot isoparametric curves
extrudedNurbsSurface.plot_isocurve_u(fig, ax, np.linspace(0, 1, 9))
extrudedNurbsSurface.plot_isocurve_v(fig, ax, np.linspace(0, 1, 5))

# Plot the base NURBS curve
nurbsCurve.plot_curve(fig, ax, color='b', linewidth=2.5)

# Show the figure
plt.show()
