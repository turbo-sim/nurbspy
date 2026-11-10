"""Join two compatible patches in the v parameter direction.

Define the neighbouring input patches, join them, and edit only the resulting
surface. Its control net and background grid remain visible.
"""

import nurbspy as nrb


def main(show=True):
    # Control points are ordered as points[u_index][v_index].
    first = nrb.NurbsSurface(
        [
            [nrb.Point(0, 0, 0), nrb.Point(1, 0, 0)],
            [nrb.Point(0, 1, 0), nrb.Point(1, 1, 0)],
        ]
    )
    second = nrb.NurbsSurface(
        [
            [nrb.Point(1, 0, 0), nrb.Point(2, 0, 0)],
            [nrb.Point(1, 1, 0), nrb.Point(2, 1, 0)],
        ]
    )
    merged = first.attach_nurbs_vdir(second)
    figure = nrb.Figure(
        merged,
        title="Surface merge in v",
        display=dict(control_points=True, control_polygon=True, grid=True),
    )
    if show:
        figure.edit()
    return figure


if __name__ == "__main__":
    main()
