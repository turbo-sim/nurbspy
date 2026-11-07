
""" Example showing how to create a ruled NURBS surface """


# -------------------------------------------------------------------------------------------------------------------- #
# Importing packages
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import nurbspy as nrb
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------------------------------------------- #
# Ruled surface example
# -------------------------------------------------------------------------------------------------------------------- #

# Define the lower NURBS curve (rational Bézier curve)
P1 = [
    nrb.Point(0.0, 0.0, 0.0),
    nrb.Point(0.25, 0.0, 0.5),
    nrb.Point(0.5, 0.0, 0.5),
    nrb.Point(0.75, 0.0, 0.0),
    nrb.Point(1.0, 0.0, 0.0),
]
W1 = np.asarray([1, 1, 2, 1, 1])
nurbsCurve1 = nrb.NurbsCurve(control_points=P1, weights=W1)

# Define the lower NURBS curve (rational Bézier curve)
P2 = [
    nrb.Point(0.0, 1.0, 0.5),
    nrb.Point(0.25, 1.0, 0.0),
    nrb.Point(0.5, 1.0, 0.0),
    nrb.Point(0.75, 1.0, 0.5),
    nrb.Point(1.0, 1.0, 0.5),
]
W2 = np.asarray([1, 1, 2, 1, 1])
nurbsCurve2 = nrb.NurbsCurve(control_points=P2, weights=W2)

# Create and plot the ruled NURBS surface
ruledNurbsSurface = nrb.NurbsSurfaceRuled(nurbsCurve1, nurbsCurve2).NurbsSurface
fig, ax = ruledNurbsSurface.plot(surface=True, surface_color='red', control_points=True)

# Plot isoparametric curves
ruledNurbsSurface.plot_isocurve_u(fig, ax, np.linspace(0, 1, 5))
ruledNurbsSurface.plot_isocurve_v(fig, ax, np.linspace(0, 1, 5))

# Plot the upper and lower NURBS curves
nurbsCurve1.plot_curve(fig, ax, color='b', linewidth=2.5)
nurbsCurve2.plot_curve(fig, ax, color='g', linewidth=2.5)



# -------------------------------------------------------------------------------------------------------------------- #
# Ruled surface example
# -------------------------------------------------------------------------------------------------------------------- #

# Create the first circular arc
O = nrb.Point(0.0, 0.0, 0.0)
X = np.asarray([1.00, 0.00, 0.00])      # Abscissa direction
Y = np.asarray([0.00, 1.00, 0.00])      # Ordinate direction
R = 0.5                                 # Circle radius
theta_start = 0.00                      # Start angle
theta_end = 2*np.pi                     # End angle
nurbsCurve1 = nrb.CircularArc(O, X, Y , R, theta_start, theta_end).NurbsCurve

# Create the second circular arc
O = nrb.Point(0.0, 0.0, 0.0)
X = np.asarray([1.00, 0.00, 0.00])      # Abscissa direction
Y = np.asarray([0.00, 0.00, 1.00])      # Ordinate direction
R = 0.5                                 # Circle radius
theta_start = np.pi/2                   # Start angle
theta_end = 2*np.pi+np.pi/2             # End angle
nurbsCurve2 = nrb.CircularArc(O, X, Y , R, theta_start, theta_end).NurbsCurve

# Create and plot the ruled NURBS surface
ruledNurbsSurface = nrb.NurbsSurfaceRuled(nurbsCurve1, nurbsCurve2).NurbsSurface
fig, ax = ruledNurbsSurface.plot(surface=False, surface_color='red', control_points=False)

# Plot isoparametric curves
ruledNurbsSurface.plot_isocurve_u(fig, ax, np.linspace(0, 1, 30))
# ruledNurbsSurface.plot_isocurve_v(fig, ax, np.linspace(0, 1, 5))

# Plot the upper and lower NURBS curves
nurbsCurve1.plot_curve(fig, ax, color='b', linewidth=2.5)
nurbsCurve2.plot_curve(fig, ax, color='g', linewidth=2.5)
plt.show()
