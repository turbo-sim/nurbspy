"""Represent an exact quarter circle with a rational quadratic Bezier."""
import numpy as np
import matplotlib.pyplot as plt
import nurbspy as nrb

nrb.set_plot_options()

P = np.array([[1., 1., 0.],
              [0., 1., 1.]])
W = np.array([1., 1 / np.sqrt(2), 1.])
# Without a degree or knot vector, these inputs define rational Bezier.
arc = nrb.NurbsCurve(control_points=P, weights=W)
polynomial = nrb.NurbsCurve(control_points=P)
u = np.linspace(0., 1., 301)
C = arc.get_value(u)
radius_error = np.max(np.abs(np.linalg.norm(C, axis=0) - 1.))
curvature = arc.get_curvature(u)

print(f"Maximum radius error: {radius_error:.2e}")
print(f"Maximum curvature error: {np.max(np.abs(curvature - 1.)):.2e}")
print("Midpoint:", arc.get_value(0.5)[:, 0].round(8))

fig, ax = plt.subplots(figsize=(5.5, 5), layout="constrained")
arc.plot_control_points(fig, ax, color="0.6")
ax.lines[-1].set_label("Control polygon")
polynomial.plot_curve(fig, ax, linestyle="--", color=nrb.COLORS_MATLAB[1])
ax.lines[-1].set_label("Polynomial quadratic")
arc.plot_curve(fig, ax, linewidth=2, color=nrb.COLORS_MATLAB[0])
ax.lines[-1].set_label("Rational quadratic: exact circle")
ax.set(xlabel="x", ylabel="y", title="An exact unit quarter circle")
ax.set_aspect("equal", adjustable="box")
ax.legend(fontsize=12)
ax.grid(alpha=0.2)
plt.show()
