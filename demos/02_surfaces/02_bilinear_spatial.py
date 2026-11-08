"""Construct and edit one warped bilinear patch from four corner points.

Show the control net and live parameter curves on a reference grid.
Drag the four control points to explore the patch shape.
"""

import numpy as np
import nurbspy as nrb


def main(show=True):
    # Bilinear patch example: Hyperbolic paraboloid

    # Set the bilinear surface defining points
    P00 = nrb.Point(0.00, 0.00, 0.00)
    P01 = nrb.Point(1.00, 0.00, 1.00)
    P10 = nrb.Point(0.00, 1.00, 1.00)
    P11 = nrb.Point(1.00, 1.00, 0.00)

    bilinearNurbsSurface = nrb.NurbsSurfaceBilinear(P00, P01, P10, P11).NurbsSurface
    figure = nrb.Figure(
        bilinearNurbsSurface,
        title="Spatial bilinear patch",
        display=dict(
            control_points=True,
            control_polygon=True,
            grid=True,
            isocurves=True,
            geometry_color="#0072bd",
        ),
    )
    if show:
        figure.edit()
    return figure


if __name__ == "__main__":
    main()
