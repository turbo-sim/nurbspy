"""Project external points onto a NURBS curve by minimizing distance to it."""
import numpy as np
import matplotlib.pyplot as plt
import nurbspy as nrb

nrb.set_plot_options()

P = np.array([[0., 1., 2., 3., 4.],
              [0., 2., -1., 2., 0.]])
curve = nrb.NurbsCurve(control_points=P, degree=3)

targets = np.array([[0.5, 3.5, 1.0],
                     [2.5, 2.5, -1.0]])

fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
curve.plot_curve(fig, ax, color=nrb.COLORS_MATLAB[0], linewidth=2)
ax.lines[-1].set_label("Curve")

for i in range(targets.shape[1]):
    target = targets[:, i]
    # maxiter, ftol, and gtol are optional overrides for the underlying
    # L-BFGS-B solve; shown here at their defaults.
    u = curve.project_point_to_curve(target, maxiter=100, ftol=1e-6, gtol=1e-6)
    foot = curve.get_value(u)[:, 0]
    tangent = curve.get_derivative(u, 1)[:, 0]
    orthogonality = np.dot(foot - target, tangent)
    print(f"Target {target}: u={u:.6f}, foot={foot.round(6)}, "
          f"(C(u)-P)*C'(u)={orthogonality:.2e}")
    ax.plot([target[0], foot[0]], [target[1], foot[1]], "k--", marker="o",
            markerfacecolor="w", linewidth=1)

ax.set(xlabel="x", ylabel="y", title="Point projection onto a NURBS curve")
ax.set_aspect("equal", adjustable="box")
ax.grid(alpha=0.2)
ax.legend()
plt.show()
