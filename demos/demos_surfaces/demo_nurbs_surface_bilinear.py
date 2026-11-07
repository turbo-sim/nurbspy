
""" Example showing how to create a bilinear NURBS surface """


# -------------------------------------------------------------------------------------------------------------------- #
# Importing packages
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import nurbspy as nrb
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------------------------------------------- #
# Bilinear patch example: Plane
# -------------------------------------------------------------------------------------------------------------------- #

# Set the bilinear surface defining points
P00 = nrb.Point(0.00, 0.00, 0.00)
P01 = nrb.Point(2.00, 0.50, 0.00)
P10 = nrb.Point(0.20, 1.00, 0.00)
P11 = nrb.Point(1.80, 1.50, 0.00)

# Create and plot the NURBS surface
bilinearNurbsSurface = nrb.NurbsSurfaceBilinear(P00, P01, P10, P11).NurbsSurface
fig, ax = bilinearNurbsSurface.plot(surface=True, surface_color='blue', control_points=True)

# Plot isoparametric curves
bilinearNurbsSurface.plot_isocurve_u(fig, ax, np.linspace(0, 1, 5))
bilinearNurbsSurface.plot_isocurve_v(fig, ax, np.linspace(0, 1, 5))


# -------------------------------------------------------------------------------------------------------------------- #
# Bilinear patch example: Hyperbolic paraboloid
# -------------------------------------------------------------------------------------------------------------------- #

# Set the bilinear surface defining points
P00 = nrb.Point(0.00, 0.00, 0.00)
P01 = nrb.Point(1.00, 0.00, 1.00)
P10 = nrb.Point(0.00, 1.00, 1.00)
P11 = nrb.Point(1.00, 1.00, 0.00)

# Create and plot the NURBS surface
bilinearNurbsSurface = nrb.NurbsSurfaceBilinear(P00, P01, P10, P11).NurbsSurface
fig, ax = bilinearNurbsSurface.plot(surface=True, surface_color='blue', control_points=True)

# Plot isoparametric curves
bilinearNurbsSurface.plot_isocurve_u(fig, ax, np.linspace(0, 1, 5))
bilinearNurbsSurface.plot_isocurve_v(fig, ax, np.linspace(0, 1, 5))

# Show the surface
plt.show()



