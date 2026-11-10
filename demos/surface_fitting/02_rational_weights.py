"""Fit control coordinates and weights at known, fixed point parameters.

This uses the same rational target and flat starting patch as demos 01, 03 and
04. Each sample retains its generating (u, v) labels, which never change during
the solve: there is no point projection. Pounce interior point fits the control
coordinates and positive weights using an exact sparse JAX Hessian. The console
prints the native convergence history and the viewer overlays the blue fitted
surface with the original rational surface in transparent orange.
The fitted control net is 16 x 16; the 2,025 target samples appear as dark dots.
"""

import numpy as np
import nurbspy as nrb

CONTROL_POINTS_PER_AXIS = 16
SAMPLES_PER_AXIS = 45
MAX_ITERATIONS = 300
TOLERANCE = 1e-8


def main(*, show=True):
    """Run one fit, optionally open its viewer, and return (result, figure)."""
    # 1. Define the same original rational patch used by the other three demos.
    axis = np.linspace(0.0, 1.0, 4)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    z = 0.55 * np.sin(2 * np.pi * x) * np.sin(np.pi * y)
    z += 0.30 * np.sin(np.pi * x) * np.sin(2 * np.pi * y)
    weights = 1.0 + 0.7 * np.sin(np.pi * x) * np.sin(np.pi * y)
    weights += 0.2 * np.sin(2 * np.pi * x) * np.sin(np.pi * y)
    original = nrb.NurbsSurface(
        np.stack((x, y, z)), weights=weights, u_degree=3, v_degree=3
    )

    # 2. Retain the generating labels as fixed correspondences for this fit.
    sample_axis = np.linspace(0.0, 1.0, SAMPLES_PER_AXIS)
    u, v = np.meshgrid(sample_axis, sample_axis, indexing="ij")
    uv = np.column_stack((u.ravel(), v.ravel()))
    points = original.get_value(*uv.T).T
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

    # 3. Optimize coordinates and logarithmic weights; one weight fixes the gauge.
    # These weight bounds are a modelling choice for this particular example.
    result = nrb.fit_surface(
        points,
        initial,
        parameters=uv,
        optimize_weights=True,
        weight_bounds=(0.2, 5.0),
        objective_scale=1000.0,
        tolerance=TOLERANCE,
        max_iterations=MAX_ITERATIONS,
        record_history=True,
        print_convergence=True,
    )
    print(f"\nConverged: {result.success} | {result.message}")
    print(f"Training RMS: {result.rms:.6e} | IPM iterations: {result.iterations}")

    # 4. View the fitted solution and original target with independent appearance.
    figure = nrb.Figure(
        result.surface,
        resolution=64,  # Use the same display sampling for both overlay surfaces.
        name="Fitted rational surface",
        title="02 / Rational fitting with known parameters",
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
