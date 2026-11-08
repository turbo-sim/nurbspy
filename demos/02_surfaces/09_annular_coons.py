"""Construct an annular patch using circular Coons boundary curves.

Define its boundary curves and display the resulting patch with its control
net and parameter curves.
"""

import numpy as np
import nurbspy as nrb


def main(show=True):
    figures = {}
    isocurve_display = dict(
        control_points=False,
        control_polygon=False,
        geometry_color="#536d7d",
        curve_depth_test=False,
    )
    # Coons surface example: Annulus NURBS surface

    # Define the outer circle
    O = nrb.Point(0.0, 0.0, 0.0)
    X = np.asarray([1.00, 0.00, 0.00])  # Abscissa direction
    Y = np.asarray([0.00, 1.00, 0.00])  # Ordinate direction
    R = 1.00  # Circle radius
    theta_start = 0.00  # Start angle (set any arbitrary value)
    theta_end = 2 * np.pi  # End angle (set any arbitrary value)
    nurbsOuter = nrb.CircularArc(O, X, Y, R, theta_start, theta_end).NurbsCurve

    # Define the inner circle
    O = nrb.Point(0.0, 0.0, 0.0)
    X = np.asarray([1.00, 0.00, 0.00])  # Abscissa direction
    Y = np.asarray([0.00, 1.00, 0.00])  # Ordinate direction
    R = 0.50  # Circle radius
    nurbsInner = nrb.CircularArc(O, X, Y, R, theta_start, theta_end).NurbsCurve

    # Define start closing boundary
    P = [
        nrb.Point(*nurbsOuter.P[:, 0]),
        nrb.Point(*nurbsInner.P[:, 0]),
    ]
    W = np.asarray([nurbsOuter.W[0], nurbsInner.W[0]])
    nurbsWest = nrb.NurbsCurve(control_points=P, weights=W)

    # Define the west boundary (rational Bézier curve)
    P = [
        nrb.Point(*nurbsOuter.P[:, -1]),
        nrb.Point(*nurbsInner.P[:, -1]),
    ]
    W = np.asarray([nurbsOuter.W[-1], nurbsInner.W[-1]])
    nurbsEast = nrb.NurbsCurve(control_points=P, weights=W)

    coonsNurbsSurface = nrb.NurbsSurfaceCoons(
        nurbsOuter, nurbsInner, nurbsWest, nurbsEast
    ).NurbsSurface
    # coonsNurbsSurface = nrb.NurbsSurfaceRuled(nurbsOuter, nurbsInner).NurbsSurface

    figures["annulus"] = nrb.Figure(
        coonsNurbsSurface,
        title="Annular Coons patch interpolating four boundary curves",
        display=dict(
            control_points=True,
            control_polygon=True,
            grid=True,
            geometry_color="#0072bd",
        ),
    )
    for value in np.linspace(0.08, 0.92, 5):
        figures["annulus"].add(
            coonsNurbsSurface.get_isocurve_u(float(value)),
            name=f"u={value:.2f}",
            display=isocurve_display,
        )
    for value in np.linspace(0.08, 0.92, 5):
        figures["annulus"].add(
            coonsNurbsSurface.get_isocurve_v(float(value)),
            name=f"v={value:.2f}",
            display=isocurve_display,
        )
    if show:
        for figure in figures.values():
            figure.show()
    return figures


if __name__ == "__main__":
    main()
