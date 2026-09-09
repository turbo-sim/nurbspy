"""Project external points onto a NURBS surface by minimizing distance to it."""
import numpy as np
import matplotlib.pyplot as plt
import nurbspy as nrb

nrb.set_plot_options()

x, y = np.meshgrid(np.linspace(0., 3., 4), np.linspace(0., 2., 3), indexing="ij")
z = np.array([[0., 1., 0.],
              [1., 2., 1.],
              [1., 2., 1.],
              [0., 1., 0.]])
P = np.stack([x, y, z])  # Shape (3, 4, 3): coordinates, u index, v index.
surface = nrb.NurbsSurface(control_points=P)

targets = np.array([[1.5, 3.5, 0.2],
                     [1.0, 0.5, 1.8],
                     [3.0, -0.5, 0.5]])

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(projection="3d")
surface.plot_surface(fig, ax, color=nrb.COLORS_MATLAB[0], alpha=0.5)

for i in range(targets.shape[1]):
    target = targets[:, i]
    # maxiter, ftol, and gtol are optional overrides for the underlying
    # L-BFGS-B solve; shown here at their defaults.
    u, v = surface.project_point_to_surface(target, maxiter=200, ftol=1e-6, gtol=1e-6)
    foot = surface.get_value(u, v)[:, 0]
    du = surface.get_derivative(u, v, order_u=1, order_v=0)[:, 0]
    dv = surface.get_derivative(u, v, order_u=0, order_v=1)[:, 0]
    orthogonality = (np.dot(foot - target, du), np.dot(foot - target, dv))
    print(f"Target {target}: (u,v)=({u:.6f}, {v:.6f}), foot={foot.round(6)}, "
          f"(S-P)*S_u,(S-P)*S_v={orthogonality[0]:.2e}, {orthogonality[1]:.2e}")
    ax.plot([target[0], foot[0]], [target[1], foot[1]], [target[2], foot[2]],
            "k--", marker="o", markerfacecolor="w", linewidth=1)

ax.set(xlabel="x", ylabel="y", zlabel="z", title="Point projection onto a NURBS surface")
plt.show()
