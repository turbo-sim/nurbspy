"""Construct an oloid from two ruled surfaces spanning perpendicular circles.

Sample the generating boundaries and display both surface patches with straight
rulings, illustrating how moving a line between the circles generates the shape.
"""

import numpy as np
import nurbspy as nrb


def main(show=True):
    # More boundary points improve the piecewise-linear boundary geometry.
    theta = np.pi - 2 * np.pi / 3 * np.cos(np.linspace(0, np.pi, 201))
    c = np.cos(theta)
    cos_phi = np.clip(c / (1 - c), -1, 1)
    sin_phi = np.sqrt(1 - cos_phi**2)
    sin_phi[[0, -1]] = 0
    first = nrb.NurbsCurve(
        [nrb.Point(x - 0.5, y, 0) for x, y in zip(c, np.sin(theta))], degree=1
    )
    patches = []
    seconds = []
    for sign in (-1, 1):
        second = nrb.NurbsCurve(
            [nrb.Point(0.5 + x, 0, sign * y) for x, y in zip(cos_phi, sin_phi)],
            degree=1,
        )
        patches.append(nrb.NurbsSurfaceRuled(first, second).NurbsSurface)
        seconds.append(second)
    figure = nrb.Figure(
        patches[0],
        title="Oloid: boundaries and straight rulings",
        display=dict(grid=True, control_points=False, control_polygon=False),
    )
    figure.add(
        patches[1],
        name="Second patch",
        display=dict(control_points=False, control_polygon=False),
    )
    line_display = dict(
        geometry_color="#536d7d", control_points=False, control_polygon=False
    )
    for i, patch in enumerate(patches):
        for j, value in enumerate(np.linspace(0.04, 0.96, 13)):
            figure.add(
                patch.get_isocurve_u(float(value)),
                name=f"Ruling {i+1}.{j+1}",
                display=line_display,
            )
    figure.add(first, name="First generating boundary", display=line_display)
    for i, second in enumerate(seconds):
        figure.add(
            second, name=f"Second generating boundary {i+1}", display=line_display
        )
    if show:
        figure.show()
    return {"oloid": figure}


if __name__ == "__main__":
    main()
