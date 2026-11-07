"""Oloid as two ruled patches with sampled, piecewise-linear NURBS boundaries."""
import numpy as np
import nurbspy as nrb
import matplotlib.pyplot as plt

# Equal-radius circles in perpendicular planes, with centres one radius apart.
# Corresponding contact points satisfy cos(phi) = cos(theta) / (1 - cos(theta)).
theta = np.pi - 2 * np.pi / 3 * np.cos(np.linspace(0, np.pi, 201))
c = np.cos(theta)
cos_phi = np.clip(c / (1 - c), -1, 1)
sin_phi = np.sqrt(1 - cos_phi**2)
curve1 = nrb.NurbsCurve(
    [nrb.Point(x - 0.5, y, 0.) for x, y in zip(c, np.sin(theta))], degree=1)

fig, ax = None, None
for sign in (-1, 1):
    curve2 = nrb.NurbsCurve(
        [nrb.Point(0.5 + x, 0., sign * y) for x, y in zip(cos_phi, sin_phi)], degree=1)
    surface = nrb.NurbsSurfaceRuled(curve1, curve2).NurbsSurface
    fig, ax = surface.plot(fig, ax, surface_color="royalblue", Nu=201, Nv=2)
    surface.plot_isocurve_u(fig, ax, np.linspace(0, 1, 17), linewidth=0.6)
    curve2.plot_curve(fig, ax, color="darkorange", linewidth=2)
curve1.plot_curve(fig, ax, color="darkorange", linewidth=2)
ax.view_init(elev=25, azim=-55)
plt.show()
