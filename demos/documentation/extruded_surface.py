"""Extrude a full rational circle into an exact cylindrical surface."""
import numpy as np
import matplotlib.pyplot as plt
import nurbspy as nrb

nrb.set_plot_options()

circle = nrb.CircularArc(
    O=np.array([0., 0., 0.]),
    X=np.array([1., 0., 0.]),
    Y=np.array([0., 1., 0.]),
    R=1., theta_start=0., theta_end=2 * np.pi,
).NurbsCurve
surface = nrb.NurbsSurfaceExtruded(
    C=circle, D=np.array([0., 0., 1.]), d=2.
).NurbsSurface

uu, vv = np.meshgrid(np.linspace(0., 1., 41), np.linspace(0., 1., 31),
                     indexing="ij")
S = surface.get_value(uu.ravel(), vv.ravel())
H, K = surface.get_curvature(uu.ravel(), vv.ravel())

print("Control net shape:", surface.P.shape)
print(f"Degrees: ({surface.p}, {surface.q})")
print(f"Maximum cylinder radius error: {np.max(np.abs(np.hypot(S[0], S[1]) - 1.)):.2e}")
print(f"Maximum height error: {np.max(np.abs(S[2] - 2 * vv.ravel())):.2e}")
print(f"Maximum Gaussian curvature magnitude: {np.max(np.abs(K)):.2e}")
print(f"Maximum |H| error from 0.5: {np.max(np.abs(np.abs(H) - 0.5)):.2e}")
v = np.linspace(0., 1., 31)
seam_error = np.max(np.abs(surface.get_value(np.zeros_like(v), v)
                           - surface.get_value(np.ones_like(v), v)))
print(f"Maximum seam mismatch: {seam_error:.2e}")

fig, ax = surface.plot(surface_color=nrb.COLORS_MATLAB[0],
                       isocurves_u=13, isocurves_v=5, Nu=101, Nv=41)
ax.set_title("Full cylinder by extrusion")
ax.set_box_aspect((1, 1, 1))
ax.view_init(elev=25, azim=35)
# Leave room for the 3D axis labels when displaying or exporting the figure.
fig.set_size_inches(7, 6)
fig.subplots_adjust(left=0.05, right=0.85, bottom=0.15, top=0.9)
ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.labelpad = 4
plt.show()
