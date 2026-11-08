"""Construct and edit one planar Bezier curve.

Define its control points and display the control polygon on a reference grid.
Frenet-Serret frames illustrate the local curve directions.
"""

import nurbspy as nrb


def main(show=True):
    # 2D Bezier curve example

    # Define the control points
    P = [
        nrb.Point(0.2, 0.5),
        nrb.Point(0.4, 0.7),
        nrb.Point(0.8, 0.6),
        nrb.Point(0.6, 0.2),
        nrb.Point(0.4, 0.2),
    ]

    bezier2D = nrb.NurbsCurve(control_points=P)
    figure = nrb.Figure(
        bezier2D,
        title="Planar Bézier curve",
        display=dict(
            view="xy",
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
