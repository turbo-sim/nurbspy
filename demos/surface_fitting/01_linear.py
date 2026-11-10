"""Approximate a rational surface with a unit-weight B-spline surface.

The sample points have known, fixed (u, v) labels. Control coordinates are fitted
by linear least squares while all weights remain one. The rational target is the
same in demos 01-04, so this less flexible model need not reproduce it exactly.
The console reports the direct solve, then the viewer compares the blue fit with
the original surface in transparent orange. No nonlinear iterations are needed.
The fitted control net is 16 x 16; the 2,025 target samples appear as dark dots.
"""

import numpy as np
import nurbspy as nrb

CONTROL_POINTS_PER_AXIS = 16
SAMPLES_PER_AXIS = 45


def main(*, show=True):
    """Run one fit, optionally open its viewer, and return (result, figure)."""
    # 1. Define the original rational patch, independently of the fitting method.
    axis = np.linspace(0.0, 1.0, 4)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    z = 0.55 * np.sin(2 * np.pi * x) * np.sin(np.pi * y)
    z += 0.30 * np.sin(np.pi * x) * np.sin(2 * np.pi * y)
    weights = 1.0 + 0.7 * np.sin(np.pi * x) * np.sin(np.pi * y)
    weights += 0.2 * np.sin(2 * np.pi * x) * np.sin(np.pi * y)
    original = nrb.NurbsSurface(
        np.stack((x, y, z)), weights=weights, u_degree=3, v_degree=3
    )

    # 2. Sample positions and retain their generating parameter labels.
    sample_axis = np.linspace(0.0, 1.0, SAMPLES_PER_AXIS)
    u, v = np.meshgrid(sample_axis, sample_axis, indexing="ij")
    uv = np.column_stack((u.ravel(), v.ravel()))
    points = original.get_value(*uv.T).T  # Public fitting inputs have shape (N, 3).

    # 3. Supply a flat patch with the desired control net, degrees and knots.
    knots = np.r_[
        np.zeros(3), np.linspace(0.0, 1.0, CONTROL_POINTS_PER_AXIS - 2), np.ones(3)
    ]
    # Greville coordinates make this flat starting surface exactly (u, v, 0).
    greville = np.array(
        [knots[i + 1 : i + 4].mean() for i in range(CONTROL_POINTS_PER_AXIS)]
    )
    x0, y0 = np.meshgrid(greville, greville, indexing="ij")
    initial = nrb.NurbsSurface(
        np.stack((x0, y0, np.zeros_like(x0))),
        u_degree=3,
        v_degree=3,
        u_knots=knots,
        v_knots=knots,
    )
    result = nrb.fit_surface(
        points,
        initial,
        parameters=uv,
        optimize_weights=False,
        record_history=True,
        print_convergence=True,
    )
    print(f"\nConverged: {result.success} | {result.message}")
    print(f"Training RMS: {result.rms:.6e}")

    # 4. Compare the returned fit with the original geometry in one Figure.
    figure = nrb.Figure(
        result.surface,
        resolution=64,  # Use the same display sampling for both overlay surfaces.
        name="Fitted B-spline",
        title="01 / Linear surface fitting",
        display=dict(
            geometry_color="#147db3",
            control_points=False,
            control_polygon=False,
            grid=False,
        ),
    )
    figure.add(
        original,
        name="Original rational surface",
        display=dict(
            geometry_color="#ed8b23",
            opacity=0.25,
            control_points=False,
            control_polygon=False,
            grid=False,
        ),
    )
    # Scatter expects (3, N); keep target samples visible above both surfaces.
    figure.scatter(points.T, color="#263238", size=2.5, depth_test=False)
    figure.set_camera(
        position=[1.8, -1.7, 1.25], target=[0.5, 0.5, 0.0], up=[0, 0, 1], fov=35
    )
    if show:
        figure.show()
    return result, figure


if __name__ == "__main__":
    main()
