"""Plot actual curve speed and compare production length methods with exact data."""

import matplotlib.pyplot as plt
import numpy as np
import nurbspy.jax as jnrb

from .common import BLUE, ORANGE, GREEN, engineering_curve, save


def generate(output_dir):
    curve = engineering_curve()
    u = np.linspace(0., 1., 601)
    xyz = curve.get_value(u)
    speed = np.linalg.norm(curve.get_derivative(u, 1), axis=0)
    cut = 0.58
    selected = u <= cut
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.5), layout="constrained")
    axes[0].plot(*xyz, color="0.7", linewidth=2)
    axes[0].plot(*xyz[:, selected], color=BLUE, linewidth=3, label="Measured segment")
    axes[0].scatter(*curve.get_value(np.array([0., cut])), s=35, color=BLUE, zorder=4)
    axes[0].set(xlabel="$x$", ylabel="$y$", title="(a) Length follows the curve")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend(loc="upper left")
    axes[1].plot(u, speed, color=BLUE)
    axes[1].fill_between(u, speed, where=selected, color=BLUE, alpha=0.18)
    axes[1].axvline(0.5, color="0.4", linestyle=":", label="Interior knot")
    axes[1].set(xlabel="Parameter $u$", ylabel="Speed $v(u)$",
                title="(b) Length is the area under speed", ylim=(0, None))
    axes[1].legend()
    save(fig, output_dir, "curve_arclength")

    # This rational line has exact length one but an increasingly thin speed
    # layer as the ratio grows. Every estimate below calls a production API.
    ratio = 1e6
    line = jnrb.NurbsCurve(np.array([[0., 1.], [0., 0.]]), np.array([1., ratio]))
    orders = np.array([5, 9, 17, 33, 65, 129])
    fixed = np.array([float(line.get_arclength_quad_fixed(n_points=int(n))) for n in orders])
    prepared = np.array([float(jnrb.prepare_arclength(line, order=int(n)).value(line.P, line.W))
                         for n in orders])
    adaptive = float(line.get_arclength(tol=1e-9))
    ode = float(line.get_arclength_ode(tol=1e-9))
    assert abs(adaptive - 1) < 1e-7
    assert abs(ode - 1) < 1e-7
    probes = np.geomspace(1e-9, 1., 600)
    actual_speed = np.linalg.norm(np.asarray(line.get_derivative(probes, 1)), axis=0)
    exact_speed = ratio / (1 + (ratio - 1) * probes)**2
    np.testing.assert_allclose(actual_speed, exact_speed, rtol=1e-9)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7), layout="constrained")
    axes[0].loglog(probes, actual_speed, color=BLUE)
    axes[0].axvline(1 / ratio, color="0.4", linestyle=":")
    axes[0].set(xlabel="Parameter $u$", ylabel="Speed $v(u)$",
                title="(a) A straight line with a narrow speed layer")
    axes[0].text(0.96, 0.93, "$w_1/w_0=10^6$\nExact length $L=1$",
                 transform=axes[0].transAxes, ha="right", va="top")
    axes[1].loglog(orders, np.maximum(abs(fixed - 1), 1e-15), "o-", color=ORANGE,
                   label="Global Clenshaw–Curtis")
    axes[1].loglog(orders, np.maximum(abs(prepared - 1), 1e-15), "s-", color=GREEN,
                   label="Prepared Gauss–Legendre")
    axes[1].axhline(max(abs(adaptive - 1), 1e-15), color=BLUE, linestyle="--",
                    label="Adaptive, endpoint grading")
    axes[1].axhline(max(abs(ode - 1), 1e-15), color="0.35", linestyle=":", label="ODE")
    axes[1].set(xlabel="Fixed-rule nodes", ylabel="Absolute length error",
                title="(b) Node count is not an accuracy guarantee")
    axes[1].legend(loc="best", fontsize=9)
    save(fig, output_dir, "arclength_methods")
    return {"exact_stress_length": 1., "adaptive_length": adaptive, "ode_length": ode,
            "fixed_nodes": orders.tolist(), "fixed_lengths": fixed.tolist(),
            "prepared_lengths": prepared.tolist()}
