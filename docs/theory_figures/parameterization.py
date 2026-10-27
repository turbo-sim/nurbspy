"""Compare three sampling choices using the actual JAX inverse-map APIs."""

import matplotlib.pyplot as plt
import numpy as np
import nurbspy.jax as nrb
from scipy.integrate import quad

from .common import BLUE, ORANGE, GREEN, engineering_curve, save


def generate(output_dir):
    curve = engineering_curve(nrb)
    count = 15
    uniform_u = np.linspace(0., 1., count)
    u_of_s, length = curve.reparametrize_by_arclength(tol=1e-9)
    distance = np.linspace(0., float(length), count)
    uniform_s = np.asarray(u_of_s(distance))
    x = np.linspace(0., 4., count)
    uniform_x = np.asarray(curve.reparametrize_by_coordinate(dim=0)(x))
    dense = np.asarray(curve.get_value(np.linspace(0., 1., 700)))
    fig, axes = plt.subplots(1, 3, figsize=(12., 3.5), layout="constrained", sharey=True)
    for ax, parameters, color, title in zip(
        axes, (uniform_u, uniform_s, uniform_x), (BLUE, ORANGE, GREEN),
        ("(a) Equal parameter increments", "(b) Equal arc-length increments", "(c) Equal x-coordinate increments"),
    ):
        points = np.asarray(curve.get_value(parameters))
        ax.plot(*dense, color="0.65", linewidth=1.3)
        ax.plot(*points, "o", color=color, markersize=5)
        ax.set(xlabel="$x$", title=title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([0, 1, 2, 3, 4])
    axes[0].set_ylabel("$y$")
    save(fig, output_dir, "curve_parameterization")
    # Verify the plotted sampling with an independent host integration, rather
    # than declaring equal straight-line chords to be equal arc lengths.
    host_curve = engineering_curve()
    def speed(t):
        return float(np.linalg.norm(host_curve.get_derivative(t, 1)))
    lengths = []
    for a, b in zip(uniform_s[:-1], uniform_s[1:]):
        breaks = [0.5] if a < 0.5 < b else None
        lengths.append(quad(speed, a, b, points=breaks, epsabs=1e-10)[0])
    step_error = np.max(abs(np.array(lengths) - float(length) / (count - 1)))
    coordinate_error = np.max(abs(np.asarray(curve.get_value(uniform_x))[0] - x))
    assert step_error < 2e-6
    assert coordinate_error < 1e-7
    return {"length": float(length), "equal_distance_step_error": float(step_error),
            "coordinate_residual": float(coordinate_error)}
