"""Build planar and spatial Bezier curves from their control points.

Display the control polygons and Frenet-Serret frames to relate the control
points to the curve shape and its local tangent, normal, and binormal directions.
"""

import nurbspy as nrb


def main(show=True):
    figures = {}
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
    figures["planar"] = nrb.Figure(
        bezier2D,
        title="Planar Bézier curve",
        display=dict(
            view="xy",
            control_points=True,
            control_polygon=True,
            grid=False,
            geometry_color="#0072bd",
            frames=True,
            frame_count=6,
            frame_scale=0.06,
        ),
    )

    # 3D Bezier curve example

    # Define the control points
    P = [
        nrb.Point(0.0, 0.0, 0.0),
        nrb.Point(0.0, 0.3, 0.05),
        nrb.Point(0.25, 0.3, 0.3),
        nrb.Point(0.5, 0.3, -0.05),
        nrb.Point(0.5, 0.1, 0.1),
    ]

    bezier3D = nrb.NurbsCurve(control_points=P)
    figures["spatial"] = nrb.Figure(
        bezier3D,
        title="Spatial Bézier curve",
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
        for figure in figures.values():
            figure.show()
    return figures


if __name__ == "__main__":
    main()
