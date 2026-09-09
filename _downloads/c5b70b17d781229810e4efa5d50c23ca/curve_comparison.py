"""Compare Bezier, B-spline, and NURBS inputs on one control polygon."""
import numpy as np
import matplotlib.pyplot as plt
import nurbspy as nrb

nrb.set_plot_options()

# Shape (2, 5): x coordinates in row zero, y coordinates in row one.
P = np.array([[0., 0.5, 1.5, 2.5, 3.],
              [0., 1.5, 2.0, 0.5, 1.]])
p = 3
U = np.array([0., 0., 0., 0., 0.5, 1., 1., 1., 1.])
W = np.array([1., 1., 3., 1., 1.])
curves = {
    "Bezier": nrb.NurbsCurve(control_points=P),
    "B-spline": nrb.NurbsCurve(control_points=P, degree=p, knots=U),
    "NURBS": nrb.NurbsCurve(control_points=P, weights=W, degree=p, knots=U),
}

u = np.linspace(0., 1., 301)
fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
curves["Bezier"].plot_control_points(fig, ax, color="0.6")
ax.lines[-1].set_label("Control polygon")
for (label, curve), color in zip(curves.items(), nrb.COLORS_MATLAB):
    C = curve.get_value(u)
    curve.plot_curve(fig, ax, color=color, linewidth=2)
    ax.lines[-1].set_label(f"{label}, degree {curve.p}")
    midpoint = curve.get_value(0.5)[:, 0]
    print(f"{label}: degree={curve.p}, C shape={C.shape}, C(0.5)={midpoint.round(6)}")
ax.set(xlabel="x", ylabel="y", title="One control polygon, three curves")
ax.set_aspect("equal", adjustable="box")
ax.grid(alpha=0.2)
ax.legend()
plt.show()
