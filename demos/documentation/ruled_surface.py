"""Connect two quadratic Bezier curves by straight rulings."""
import numpy as np
import matplotlib.pyplot as plt
import nurbspy as nrb

nrb.set_plot_options()

P1 = np.array([[0., 1.5, 3.],
               [0., 0., 0.],
               [0., 1.5, 0.]])
P2 = np.array([[0., 1.5, 3.],
               [2., 2., 2.],
               [1., 0.2, 1.5]])
C1 = nrb.NurbsCurve(control_points=P1)
C2 = nrb.NurbsCurve(control_points=P2)
surface = nrb.NurbsSurfaceRuled(C1, C2).NurbsSurface

u = np.linspace(0., 1., 61)
v = np.full_like(u, 0.35)
expected = (1 - v) * C1.get_value(u) + v * C2.get_value(u)
error = np.max(np.abs(surface.get_value(u, v) - expected))
print("Control net shape:", surface.P.shape)
print(f"Degrees: ({surface.p}, {surface.q})")
print(f"Maximum linear-blend error: {error:.2e}")
print("Midpoint:", surface.get_value(0.5, 0.5)[:, 0].round(6))

fig, ax = surface.plot(surface_color=nrb.COLORS_MATLAB[0],
                       isocurves_u=13, isocurves_v=3)
C1.plot_curve(fig, ax, color=nrb.COLORS_MATLAB[1], linewidth=2.5)
C2.plot_curve(fig, ax, color=nrb.COLORS_MATLAB[1], linewidth=2.5)
ax.set_title("Ruled surface between two curves")
ax.view_init(elev=25, azim=-55)
# Leave room for the 3D axis labels when displaying or exporting the figure.
fig.set_size_inches(7, 6)
fig.subplots_adjust(left=0.05, right=0.85, bottom=0.15, top=0.9)
ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.labelpad = 4
plt.show()
