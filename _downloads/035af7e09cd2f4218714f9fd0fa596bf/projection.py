"""Plot curve/surface projections computed with the public single-start JAX APIs."""

import matplotlib.pyplot as plt
import numpy as np
import nurbspy.jax as nrb

from .common import BLUE, ORANGE, GREEN, engineering_curve, paraboloid, surface_grid, axis3d, save


def generate(output_dir):
    curve = engineering_curve(nrb)
    queries = np.array([[1.2, -0.5, 3.1], [1.7, -0.3, 0.9]])
    parameters = np.asarray(curve.project_points(queries, tol=1e-10))
    feet = np.asarray(curve.get_value(parameters))
    dense_u = np.linspace(0., 1., 900)
    dense = np.asarray(curve.get_value(dense_u))
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), layout="constrained")
    axes[0].plot(*dense, color="0.25", linewidth=2)
    max_interior_residual = 0.
    for i, color in enumerate((BLUE, ORANGE, GREEN)):
        axes[0].plot([queries[0, i], feet[0, i]], [queries[1, i], feet[1, i]],
                     "--", color=color, linewidth=1.2)
        axes[0].plot(*queries[:, i], "x", color=color, markersize=7, label=f"Query {i+1}")
        axes[0].plot(*feet[:, i], "o", color=color, markersize=5)
        objective = np.sum((dense - queries[:, i, None])**2, axis=0) / 2
        axes[1].plot(dense_u, objective, color=color, label=f"Query {i+1}")
        minimum = np.sum((feet[:, i] - queries[:, i])**2) / 2
        axes[1].plot(parameters[i], minimum, "o", color=color)
        if 0 < parameters[i] < 1:
            tangent = np.asarray(curve.get_tangent(parameters[i])).ravel()
            max_interior_residual = max(max_interior_residual, abs(float((feet[:, i]-queries[:, i]) @ tangent)))
        assert minimum <= objective.min() + 1e-8
    axes[0].set(xlabel="$x$", ylabel="$y$", title="(a) Interior and endpoint projections")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend(loc="upper left")
    axes[1].set(xlabel="Parameter $u$", ylabel="$F(u)=\\frac{1}{2}\\|C(u)-Q\\|^2$",
                title="(b) Minimize distance over the bounded domain", ylim=(0, None))
    save(fig, output_dir, "curve_projection")

    surface = paraboloid(nrb)
    queries3 = np.array([[0.2, -1.65, 1.7], [0.1, 0.1, 1.8], [-0.4, 0.2, 0.8]])
    uv = np.asarray(surface.project_points(queries3, gtol=1e-10))
    feet3 = np.asarray(surface.get_value(uv[0], uv[1]))
    ug, vg, xyz = surface_grid(surface, 65)
    fig = plt.figure(figsize=(10.4, 4.6), layout="constrained")
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(*xyz, color=BLUE, alpha=0.25, linewidth=0, rcount=65, ccount=65)
    for i, color in enumerate((BLUE, ORANGE, GREEN)):
        ax.plot(*np.column_stack((queries3[:, i], feet3[:, i])), "--", color=color, linewidth=1.3)
        ax.scatter(*queries3[:, i], marker="x", color=color, s=40)
        ax.scatter(*feet3[:, i], color=color, s=25)
    axis3d(ax, np.column_stack((xyz.reshape(3, -1), queries3)), elev=25, azim=-62)
    ax.set_title("(a) Closest points on a bounded surface patch")
    ax2 = fig.add_subplot(1, 2, 2)
    # Parameter-space contour lines are actual surface heights, not a sketch.
    contour = ax2.contour(ug, vg, xyz[2], levels=[0.1, 0.3, 0.5, 0.7, 0.9],
                          colors="0.7", linewidths=0.8)
    ax2.clabel(contour, fontsize=8, fmt="%.1f")
    for i, (color, label) in enumerate(zip((BLUE, ORANGE, GREEN), ("Interior", "Edge", "Corner"))):
        ax2.plot(*uv[:, i], "o", color=color, label=label, markersize=7, clip_on=False)
    ax2.set(xlabel="$u$", ylabel="$v$", title="(b) The returned parameters; contours show $z$",
            xlim=(0, 1), ylim=(0, 1))
    ax2.set_aspect("equal")
    ax2.legend(loc="center right")
    save(fig, output_dir, "surface_projection")
    assert np.all(uv[:, 0] > 0) and np.all(uv[:, 0] < 1)
    assert uv[0, 1] == 0 and 0 < uv[1, 1] < 1
    np.testing.assert_array_equal(uv[:, 2], [1., 1.])
    assert max_interior_residual < 1e-8
    return {"curve_parameters": parameters.tolist(),
            "curve_interior_normal_residual": max_interior_residual,
            "surface_parameters": uv.tolist()}
