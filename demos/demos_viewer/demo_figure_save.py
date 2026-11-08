"""Save a complete interactive figure, restore it, and export a PNG."""

from pathlib import Path
import numpy as np
import nurbspy as nrb


def make_figure():
    x, y = np.meshgrid(np.linspace(-1.5, 1.5, 4), np.linspace(-1.5, 1.5, 4), indexing="ij")
    surface = nrb.NurbsSurface([[nrb.Point(x[i, j], y[i, j], .3 * (x[i, j] ** 2 - y[i, j] ** 2))
                                for j in range(4)] for i in range(4)])
    figure = nrb.Figure(surface, name="saddle", title="Reproducible surface figure",
                        display=dict(control_points=False, control_polygon=False))
    figure.add(nrb.NurbsCurve([nrb.Point(-1.5, 0, .7), nrb.Point(1.5, 0, .7)]),
               name="reference line", display=dict(geometry_color="#ff7f0e", control_points=False, control_polygon=False))
    return figure


if __name__ == "__main__":
    output = Path("output/figures")
    output.mkdir(parents=True, exist_ok=True)
    figure = make_figure()
    figure.show()  # Adjust the camera and per-object settings, then close.
    figure.to_yaml(output / "surface_view.yaml")
    restored = nrb.Figure.from_yaml(output / "surface_view.yaml")
    restored.savefig(output / "surface.png", width=2400, height=1800)
