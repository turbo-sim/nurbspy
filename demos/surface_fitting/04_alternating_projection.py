"""Fit an unlabeled cloud by alternating surface fits and point projection.

This uses the same target, flat start, bounds and regularization as demo 03.
Each surface step holds the point parameters fixed and uses Pounce interior point
with an exact sparse JAX Hessian. Each projection step holds the geometry fixed
and reuses NurbsPy's existing bounded-Newton projector. Generating labels do not
enter the fit. The console prints the convergence history and outer-cycle checks,
then the viewer shows the fit in blue and the original in transparent orange.
An exhausted cycle budget is reported as unsuccessful, even if inner fits converge.
The fitted control net is 16 x 16; the 2,025 target samples appear as dark dots.
"""

import numpy as np
import nurbspy as nrb

CONTROL_POINTS_PER_AXIS = 16
SAMPLES_PER_AXIS = 45
MAX_ITERATIONS = 300
MAX_CYCLES = 100
TOLERANCE = 1e-8


def main(*, show=True):
    """Run one fit, optionally open its viewer, and return (result, figure)."""
    # 1. Construct the original rational patch used in all four demos.
    axis = np.linspace(0.0, 1.0, 4)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    z = 0.55 * np.sin(2 * np.pi * x) * np.sin(np.pi * y)
    z += 0.30 * np.sin(np.pi * x) * np.sin(2 * np.pi * y)
    weights = 1.0 + 0.7 * np.sin(np.pi * x) * np.sin(np.pi * y)
    weights += 0.2 * np.sin(2 * np.pi * x) * np.sin(np.pi * y)
    original = nrb.NurbsSurface(
        np.stack((x, y, z)), weights=weights, u_degree=3, v_degree=3
    )

    # 2. Create the cloud and a flat initial patch; do not provide target labels.
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

    # 3. Match demo 03's objective so only the solution strategy differs.
    lower = np.full_like(initial.P, -0.25)
    upper = np.full_like(initial.P, 1.25)
    lower[2], upper[2] = -0.8, 0.8
    frame_penalty = np.zeros_like(initial.P)
    frame_penalty[:2] = 1e-6
    result = nrb.fit_surface(
        points,
        initial,
        correspondence="alternating",
        optimize_weights=True,
        control_bounds=(lower, upper),
        weight_bounds=(0.2, 5.0),
        control_regularization=frame_penalty,
        weight_regularization=1e-9,
        objective_scale=1000.0,
        tolerance=TOLERANCE,
        max_iterations=MAX_ITERATIONS,
        max_cycles=MAX_CYCLES,
        record_history=True,
        print_convergence=True,
    )
    print(f"\nConverged: {result.success} | {result.message}")
    print(
        f"Training RMS: {result.rms:.6e} | Fitting IPM iterations: {result.iterations}"
    )

    # 4. Show the actual returned geometry, including budget-limited results.
    figure = nrb.Figure(
        result.surface,
        resolution=64,  # Use the same display sampling for both overlay surfaces.
        name="Alternating fitted surface",
        title="04 / Alternating projection and fitting",
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
