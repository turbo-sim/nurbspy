"""Fit an unlabeled point cloud by optimizing the surface and parameters jointly.

The target is the same rational patch as in demos 01, 02 and 04. Its generating
(u, v) labels are used only to create the cloud; the fitting problem receives
positions alone. Initial guesses come from projecting onto the flat starting
patch. Pounce interior point then changes control coordinates, positive weights
and all point parameters together, with the full exact sparse JAX Hessian.
Native convergence is printed before showing the fit in blue and the original
surface in transparent orange. This is a joint solve, not projection/refitting.
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
    # 1. Build the same reference geometry used by the other three methods.
    axis = np.linspace(0.0, 1.0, 4)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    z = 0.55 * np.sin(2 * np.pi * x) * np.sin(np.pi * y)
    z += 0.30 * np.sin(np.pi * x) * np.sin(2 * np.pi * y)
    weights = 1.0 + 0.7 * np.sin(np.pi * x) * np.sin(np.pi * y)
    weights += 0.2 * np.sin(2 * np.pi * x) * np.sin(np.pi * y)
    original = nrb.NurbsSurface(
        np.stack((x, y, z)), weights=weights, u_degree=3, v_degree=3
    )

    # 2. Sample the target, then pass only positions to the fitting problem.
    sample_axis = np.linspace(0.0, 1.0, SAMPLES_PER_AXIS)
    u, v = np.meshgrid(sample_axis, sample_axis, indexing="ij")
    points = original.get_value(u.ravel(), v.ravel()).T
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

    # 3. Choose bounds and weak frame/weight penalties for this unit-sized patch.
    # The shared fitter has no hard-coded terrain bounds or x/y initialization.
    lower = np.full_like(initial.P, -0.25)
    upper = np.full_like(initial.P, 1.25)
    lower[2], upper[2] = -0.8, 0.8
    frame_penalty = np.zeros_like(initial.P)
    frame_penalty[:2] = 1e-6
    result = nrb.fit_surface(
        points,
        initial,
        correspondence="joint",
        optimize_weights=True,
        control_bounds=(lower, upper),
        weight_bounds=(0.2, 5.0),
        control_regularization=frame_penalty,
        weight_regularization=1e-9,
        objective_scale=1000.0,
        tolerance=TOLERANCE,
        max_iterations=MAX_ITERATIONS,
        record_history=True,
        print_convergence=True,
    )
    print(f"\nConverged: {result.success} | {result.message}")
    print(f"Training RMS: {result.rms:.6e} | IPM iterations: {result.iterations}")

    # 4. The fit can use a different parameterization; compare its geometry in 3D.
    figure = nrb.Figure(
        result.surface,
        resolution=64,  # Use the same display sampling for both overlay surfaces.
        name="Jointly fitted surface",
        title="03 / Joint surface and parameter fitting",
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
