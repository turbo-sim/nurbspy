"""Join compatible NURBS patches along either parameter direction.

Display the two inputs and the merged patch together. Raise the result
slightly to separate it visually from the input patches.
"""

import nurbspy as nrb


def main(show=True):
    figures = {}
    # Create the first NURBS surface

    # Define the control points
    n_dim, n, m = 3, 2, 2
    P = [
        [nrb.Point(0.0, 0.0, 0.0), nrb.Point(1.0, 0.0, 0.0)],
        [nrb.Point(0.0, 1.0, 0.0), nrb.Point(1.0, 1.0, 0.0)],
    ]  # points[u_index][v_index]

    nurbsSurface1 = nrb.NurbsSurface(control_points=P)

    # Create the second NURBS surface

    # Define the control points
    n_dim, n, m = 3, 2, 2
    P = [
        [nrb.Point(0.0, 1.0, 0.0), nrb.Point(1.0, 1.0, 0.0)],
        [nrb.Point(0.0, 2.0, 0.0), nrb.Point(1.0, 2.0, 0.0)],
    ]  # points[u_index][v_index]

    nurbsSurface2 = nrb.NurbsSurface(control_points=P)

    # Create the third NURBS surface

    # Define the control points
    n_dim, n, m = 3, 2, 2
    P = [
        [nrb.Point(1.0, 0.0, 0.0), nrb.Point(2.0, 0.0, 0.0)],
        [nrb.Point(1.0, 1.0, 0.0), nrb.Point(2.0, 1.0, 0.0)],
    ]  # points[u_index][v_index]

    nurbsSurface3 = nrb.NurbsSurface(control_points=P)

    # Merge two NURBS surfaces with a common u-side

    # Merge the surfaces
    nurbsMerged_u = nurbsSurface1.attach_nurbs_udir(nurbsSurface2)
    nurbsMerged_u.P[2, :, :] = 0.5

    # Merge two NURBS surfaces with a common v-side

    # Merge the surfaces
    nurbsMerged_v = nurbsSurface1.attach_nurbs_vdir(nurbsSurface3)
    nurbsMerged_v.P[2, :, :] = 0.5
    # The joined patch is raised by 0.5 only to separate it visually from its inputs.
    for direction, second, merged in (
        ("u", nurbsSurface2, nurbsMerged_u),
        ("v", nurbsSurface3, nurbsMerged_v),
    ):
        fig = nrb.Figure(
            nurbsSurface1,
            title=f"Joining surfaces in {direction}: raised result",
            display=dict(
                control_points=True,
                control_polygon=True,
                grid=True,
                geometry_color="#0072bd",
                opacity=0.4,
            ),
        )
        fig.add(
            second,
            name="Second input",
            display=dict(geometry_color="#21856a", opacity=0.4),
        )
        fig.add(
            merged,
            name="Joined patch (raised for visibility)",
            display=dict(control_points=True, control_polygon=True),
        )
        figures[f"merged_{direction}"] = fig
    if show:
        for figure in figures.values():
            figure.show()
    return figures


if __name__ == "__main__":
    main()
