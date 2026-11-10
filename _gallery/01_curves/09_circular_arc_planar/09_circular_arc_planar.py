"""Represent and edit one planar circular arc with rational NURBS.

Check curvature, torsion, and arc length against the exact circular values
before opening the editor. Moving control points can change the circular shape.
"""

import numpy as np
import nurbspy as nrb


def main(show=True):
    # 2D circular arc example

    O = nrb.Point(0.0, 1.0)
    X = np.array([1.00, 0.00])  # X-direction in circle plane
    Y = np.array([0.00, 1.00])  # Y-direction in circle plane
    R = 0.5  # Radius
    theta_start = 1 / 6 * np.pi
    theta_end = 3 / 2 * np.pi - 1 / 6 * np.pi

    circ2D = nrb.CircularArc(O, X, Y, R, theta_start, theta_end)
    figure = nrb.Figure(
        circ2D.NurbsCurve,
        title="Planar circular arc",
        display=dict(
            view="xy",
            control_points=True,
            control_polygon=True,
            grid=True,
            geometry_color="#0072bd",
        ),
    )

    # Evaluate along the curve
    u = np.linspace(0, 1, 100)
    curv = circ2D.NurbsCurve.get_curvature(u)
    tors = circ2D.NurbsCurve.get_torsion(u)
    # Integrate each knot span separately, avoiding quadrature across span joins.
    breaks = np.unique(circ2D.NurbsCurve.U)
    arc_len = sum(
        circ2D.NurbsCurve.get_arclength(a, b) for a, b in zip(breaks[:-1], breaks[1:])
    )

    # Analytical values
    theta_span = np.abs(theta_end - theta_start)
    curv_exact = 1 / R
    tors_exact = 0.0
    arc_exact = R * theta_span

    # Report results
    print("\n=== 2D circular arc check ===")
    print(f"Expected curvature          : {curv_exact:.6f}")
    print(
        f"Curvature RMS error         : {np.sqrt(np.mean((curv - curv_exact)**2)):.3e}"
    )
    print(
        f"Torsion RMS error           : {np.sqrt(np.mean((tors - tors_exact)**2)):.3e}"
    )
    print(f"Arc length (computed)       : {arc_len:.6f}")
    print(f"Arc length (analytical)     : {arc_exact:.6f}")
    print(f"Arc length absolute error   : {np.abs(arc_len - arc_exact):.3e}")
    if show:
        figure.edit()
    return figure


if __name__ == "__main__":
    main()
