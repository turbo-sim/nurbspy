"""Generate surfaces by extruding a profile along a fixed direction.

Extrude a NURBS profile and an exact circular arc to form a shaped patch and a
cylinder. Show the generating curves, parameter curves, and control nets.
"""

import numpy as np
import nurbspy as nrb


def main(show=True):
    figures = {}
    boundary_display = dict(
        geometry_color="#d97823",
        control_points=False,
        control_polygon=False,
        curve_depth_test=False,
    )
    isocurve_display = dict(
        control_points=False,
        control_polygon=False,
        geometry_color="#536d7d",
        curve_depth_test=False,
    )
    # Extruded surface example:

    # Define the base NURBS curve (rational Bézier curve)
    P = [
        nrb.Point(0.0, 0.0, 0.0),
        nrb.Point(0.5, 0.8, 0.0),
        nrb.Point(0.75, 0.6, 0.0),
        nrb.Point(1.0, 0.3, 0.0),
        nrb.Point(0.8, 0.1, 0.0),
    ]
    W = np.asarray([1, 1, 1, 1, 1])
    nurbsCurve = nrb.NurbsCurve(control_points=P, weights=W)

    # Set the extrusion direction (can be unitary or not)
    direction = np.asarray([1, 1, 5])

    # Set the extrusion length
    length = 1

    extrudedNurbsSurface = nrb.NurbsSurfaceExtruded(
        nurbsCurve, direction, length
    ).NurbsSurface
    figures["profile_extrusion"] = nrb.Figure(
        extrudedNurbsSurface,
        title="Extruded rational profile",
        display=dict(
            control_points=True,
            control_polygon=True,
            grid=True,
            geometry_color="#0072bd",
        ),
    )
    for value in np.linspace(0.08, 0.92, 5):
        figures["profile_extrusion"].add(
            extrudedNurbsSurface.get_isocurve_u(float(value)),
            name=f"u={value:.2f}",
            display=isocurve_display,
        )
    for value in np.linspace(0.08, 0.92, 5):
        figures["profile_extrusion"].add(
            extrudedNurbsSurface.get_isocurve_v(float(value)),
            name=f"v={value:.2f}",
            display=isocurve_display,
        )
    figures["profile_extrusion"].add(
        nurbsCurve,
        name="Generating curve",
        display=boundary_display,
    )

    # Extruded surface example: A circular cylinder

    # Define a circular arc
    O = nrb.Point(0.0, 0.0, 0.0)
    X = np.asarray([1.00, 0.00, 0.00])  # Abscissa direction
    Y = np.asarray([0.00, 1.00, 0.00])  # Ordinate direction
    R = 0.5  # Circle radius
    theta_start = 0.00  # Start angle
    theta_end = 2 * np.pi  # End angle

    nurbsCurve = nrb.CircularArc(O, X, Y, R, theta_start, theta_end).NurbsCurve

    # Set the extrusion direction (can be unitary or not)
    direction = np.asarray([0, 0, 1])

    # Set the extrusion length
    length = 1

    extrudedNurbsSurface = nrb.NurbsSurfaceExtruded(
        nurbsCurve, direction, length
    ).NurbsSurface
    figures["cylinder"] = nrb.Figure(
        extrudedNurbsSurface,
        title="Circular cylinder by extrusion",
        display=dict(
            control_points=True,
            control_polygon=True,
            grid=True,
            geometry_color="#0072bd",
        ),
    )
    for value in np.linspace(0.08, 0.92, 5):
        figures["cylinder"].add(
            extrudedNurbsSurface.get_isocurve_u(float(value)),
            name=f"u={value:.2f}",
            display=isocurve_display,
        )
    for value in np.linspace(0.08, 0.92, 5):
        figures["cylinder"].add(
            extrudedNurbsSurface.get_isocurve_v(float(value)),
            name=f"v={value:.2f}",
            display=isocurve_display,
        )
    figures["cylinder"].add(
        nurbsCurve,
        name="Generating curve",
        display=boundary_display,
    )
    if show:
        for figure in figures.values():
            figure.show()
    return figures


if __name__ == "__main__":
    main()
