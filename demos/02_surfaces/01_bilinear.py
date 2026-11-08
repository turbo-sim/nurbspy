"""Create bilinear NURBS patches from four corner points.

Compare planar and warped patches, showing their control nets and parameter
curves to illustrate interpolation between the corners.
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
    # Bilinear patch example: Plane

    # Set the bilinear surface defining points
    P00 = nrb.Point(0.00, 0.00, 0.00)
    P01 = nrb.Point(2.00, 0.50, 0.00)
    P10 = nrb.Point(0.20, 1.00, 0.00)
    P11 = nrb.Point(1.80, 1.50, 0.00)

    bilinearNurbsSurface = nrb.NurbsSurfaceBilinear(P00, P01, P10, P11).NurbsSurface
    figures["planar"] = nrb.Figure(
        bilinearNurbsSurface,
        title="Planar bilinear patch",
        display=dict(
            view="xy",
            control_points=True,
            control_polygon=True,
            grid=False,
            geometry_color="#0072bd",
        ),
    )
    for value in np.linspace(0.08, 0.92, 5):
        figures["planar"].add(
            bilinearNurbsSurface.get_isocurve_u(float(value)),
            name=f"u={value:.2f}",
            display=isocurve_display,
        )
    for value in np.linspace(0.08, 0.92, 5):
        figures["planar"].add(
            bilinearNurbsSurface.get_isocurve_v(float(value)),
            name=f"v={value:.2f}",
            display=isocurve_display,
        )

    # Bilinear patch example: Hyperbolic paraboloid

    # Set the bilinear surface defining points
    P00 = nrb.Point(0.00, 0.00, 0.00)
    P01 = nrb.Point(1.00, 0.00, 1.00)
    P10 = nrb.Point(0.00, 1.00, 1.00)
    P11 = nrb.Point(1.00, 1.00, 0.00)

    bilinearNurbsSurface = nrb.NurbsSurfaceBilinear(P00, P01, P10, P11).NurbsSurface
    figures["spatial"] = nrb.Figure(
        bilinearNurbsSurface,
        title="Spatial bilinear patch",
        display=dict(
            control_points=True,
            control_polygon=True,
            grid=True,
            geometry_color="#0072bd",
        ),
    )
    for value in np.linspace(0.08, 0.92, 5):
        figures["spatial"].add(
            bilinearNurbsSurface.get_isocurve_u(float(value)),
            name=f"u={value:.2f}",
            display=isocurve_display,
        )
    for value in np.linspace(0.08, 0.92, 5):
        figures["spatial"].add(
            bilinearNurbsSurface.get_isocurve_v(float(value)),
            name=f"v={value:.2f}",
            display=isocurve_display,
        )
    if show:
        for figure in figures.values():
            figure.show()
    return figures


if __name__ == "__main__":
    main()
