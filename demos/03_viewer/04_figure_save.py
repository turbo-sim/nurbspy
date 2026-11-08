"""Save a complete interactive figure, restore it, and export a PNG."""

from pathlib import Path
import numpy as np
import nurbspy as nrb


def make_figure():
    # A 4 x 4 control net defines the saddle's shape, not its image pixel count.
    x, y = np.meshgrid(
        np.linspace(-1.5, 1.5, 4), np.linspace(-1.5, 1.5, 4), indexing="ij"
    )
    surface = nrb.NurbsSurface(
        [
            [
                nrb.Point(x[i, j], y[i, j], 0.3 * (x[i, j] ** 2 - y[i, j] ** 2))
                for j in range(4)
            ]
            for i in range(4)
        ]
    )
    # Figure automatically chooses display sampling from the surface shape.
    # Its default logical size is 1000 x 750 pixels, fixing a 4:3 composition.
    # Hide the control points and their connecting net for an uncluttered view.
    figure = nrb.Figure(
        surface,
        name="saddle",
        title="Reproducible surface figure",
        display=dict(control_points=False, control_polygon=False),
    )
    # Add an independently coloured reference line to the same scene and camera.
    figure.add(
        nrb.NurbsCurve([nrb.Point(-1.5, 0, 0.7), nrb.Point(1.5, 0, 0.7)]),
        name="reference line",
        display=dict(
            geometry_color="#ff7f0e", control_points=False, control_polygon=False
        ),
    )
    return figure


if __name__ == "__main__":
    output = Path(__file__).resolve().parent / "output" / Path(__file__).stem
    output.mkdir(parents=True, exist_ok=True)
    figure = make_figure()
    figure.show()  # Adjust the camera and per-object settings, then close.
    # YAML stores the geometry, camera, and appearance so the view can be restored.
    figure.to_yaml(output / "surface_view.yaml")
    restored = nrb.Figure.from_yaml(output / "surface_view.yaml")
    # width/height are PNG pixels, separate from geometry sampling resolution.
    # 2400 x 1800 keeps the saved 4:3 composition and gives a high-resolution image.
    restored.savefig(output / "surface.png", width=2400, height=1800)
