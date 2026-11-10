"""Construct and edit one planar bilinear patch from four corner points.

Show the control net and live parameter curves on a reference grid.
Drag the four control points to explore the patch shape.
"""

import numpy as np
import nurbspy as nrb


def main(show=True):
    # Bilinear patch example: Plane

    # Set the bilinear surface defining points
    P00 = nrb.Point(0.00, 0.00, 0.00)
    P01 = nrb.Point(2.00, 0.50, 0.00)
    P10 = nrb.Point(0.20, 1.00, 0.00)
    P11 = nrb.Point(1.80, 1.50, 0.00)

    bilinearNurbsSurface = nrb.NurbsSurfaceBilinear(P00, P01, P10, P11).NurbsSurface
    figure = nrb.Figure(
        bilinearNurbsSurface,
        title="Planar bilinear patch",
        display=dict(
            view="xy",
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
