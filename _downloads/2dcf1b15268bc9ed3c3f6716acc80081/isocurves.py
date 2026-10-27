"""Extract actual rational section curves from a torus and check equality."""

import matplotlib.pyplot as plt
import numpy as np

from .common import BLUE, ORANGE, GREEN, torus, surface_grid, axis3d, save


def generate(output_dir):
    surface = torus()
    _, _, xyz = surface_grid(surface)
    samples = np.linspace(0., 1., 401)
    fixed = [0.12, 0.35, 0.62]
    fig = plt.figure(figsize=(10.5, 4.6), layout="constrained")
    max_error = 0.
    for column, direction in enumerate(("u", "v"), start=1):
        ax = fig.add_subplot(1, 2, column, projection="3d")
        ax.plot_surface(*xyz, color="0.78", alpha=0.2, linewidth=0, shade=False,
                        rcount=61, ccount=61)
        for value, color in zip(fixed, (BLUE, ORANGE, GREEN)):
            if direction == "u":
                section = surface.get_isocurve_u(value)
                reference = surface.get_value(np.full_like(samples, value), samples)
            else:
                section = surface.get_isocurve_v(value)
                reference = surface.get_value(samples, np.full_like(samples, value))
            points = section.get_value(samples)
            max_error = max(max_error, float(np.max(abs(points - reference))))
            section.plot_curve(fig, ax, color=color, linewidth=2.3)
            # Add labels through proxy handles to keep 3D legends portable.
            ax.plot([], [], [], color=color, label=f"${direction}_0={value:.2f}$")
        axis3d(ax, xyz, elev=30, azim=-52)
        ax.set_title(f"({chr(96 + column)}) Fixed ${direction}$; the other parameter varies")
        ax.legend(loc="upper left", fontsize=9)
    save(fig, output_dir, "surface_isocurves")
    assert max_error < 1e-12
    return {"section_surface_max_error": max_error}
