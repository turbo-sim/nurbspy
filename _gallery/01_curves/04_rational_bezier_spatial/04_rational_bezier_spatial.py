"""Construct and edit one spatial rational Bezier curve.

Define its control points and display the control polygon on a reference grid.
Frenet-Serret frames illustrate the local curve directions.
"""

import numpy as np
import nurbspy as nrb


def main(show=True):
    # 3D rational Bezier curve example

    # Define the control points
    P = [
        nrb.Point(0.0, 0.0, 0.0),
        nrb.Point(0.0, 0.3, 0.05),
        nrb.Point(0.25, 0.3, 0.3),
        nrb.Point(0.5, 0.3, -0.05),
        nrb.Point(0.5, 0.1, 0.1),
    ]

    # Define the array of control point weights
    W = np.asarray([1, 1, 3, 1, 1])

    bezier3D = nrb.NurbsCurve(control_points=P, weights=W)
    figure = nrb.Figure(
        bezier3D,
        title="Rational Spatial Bézier curve",
        display=dict(
            control_points=True,
            control_polygon=True,
            grid=True,
            geometry_color="#0072bd",
            frames=True,
            frame_count=6,
            frame_scale=0.06,
        ),
    )
    if show:
        figure.edit()
    return figure


if __name__ == "__main__":
    main()
