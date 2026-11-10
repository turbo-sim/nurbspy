"""Public fixed-topology fitting and correspondence strategies."""

from copy import deepcopy
from dataclasses import dataclass, field
from importlib import import_module
from time import perf_counter
import numpy as np
from ._common import (
    FitLayout,
    control_penalty,
    evaluate,
    loss,
    parameter_domain,
    validate_parameters,
    validate_points,
)
from ._linear import solve_linear


@dataclass
class SurfaceFitResult:
    """Returned geometry, correspondences and explicitly reported convergence.

    ``history`` is optional. Native observations have unknown acceptance and
    identify their coordinates as last objective evaluations. The final entry
    is the authoritative returned solution. ``timings`` separates compilation,
    setup and warm solve; ``diagnostics`` retains solver and block-check metrics.
    """

    surface: object
    parameters: np.ndarray
    success: bool
    message: str
    iterations: int
    objective: float
    rms: float
    history: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


def _positive(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not np.isscalar(value)
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")


def _budget(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")


def _sum_info(totals, info):
    for key, value in info.items():
        if key.endswith("_seconds") or key in ("iterations", "nfev", "njev", "nhev"):
            totals[key] = totals.get(key, 0) + value


def _endpoints(layout, x, parameters, seconds, *, phase):
    return [
        dict(
            iteration=0,
            x=layout.x0.copy(),
            parameters=parameters.copy(),
            seconds=0.0,
            event_kind="initial",
            accepted=True,
            phase=phase,
        ),
        dict(
            iteration=0,
            x=np.asarray(x).copy(),
            parameters=parameters.copy(),
            seconds=seconds,
            event_kind="solution",
            accepted=True,
            phase=phase,
        ),
    ]


def _finish(layout, points, x, parameters, info, frames, options, preparation):
    begun = perf_counter()
    value, rms = loss(layout, x, points, parameters, **options)
    geometry = layout.surface(x)
    history = []
    for frame in frames:
        item = dict(frame)
        state = item.pop("x", None)
        if state is not None:
            uv = np.array(item.get("parameters", parameters), copy=True)
            observed, observed_rms = loss(layout, state, points, uv, **options)
            item.update(
                surface=layout.surface(state),
                parameters=uv,
                training_rms=observed_rms,
                rms=observed_rms,
                evaluation_objective=observed,
            )
            item.setdefault("objective", observed)
        history.append(item)
    info = dict(info)
    info["setup_seconds"] = info.get("setup_seconds", 0.0) + preparation
    info["postprocess_seconds"] = perf_counter() - begun
    info.setdefault("kkt_seconds", 0.0)
    info["variables"] = layout.size + (
        2 * len(points) if info.get("correspondence") != "fixed" else 0
    )
    info["surface_variables"] = layout.size
    info["objective_scale"] = options["objective_scale"]
    info["bound_violation"] = max(
        0.0,
        float(np.max(layout.lower - np.asarray(x)[: layout.size], initial=0.0)),
        float(np.max(np.asarray(x)[: layout.size] - layout.upper, initial=0.0)),
    )
    lower, upper = parameter_domain(layout.initial)
    info["parameter_violation"] = max(
        0.0, float(np.max(lower - parameters)), float(np.max(parameters - upper))
    )
    feasibility_tolerance = info.get("tolerance", info.get("native_tolerance", 1e-8))
    info["feasibility_success"] = (
        max(info["bound_violation"], info["parameter_violation"])
        <= feasibility_tolerance
    )
    if not info["feasibility_success"]:
        info.update(
            success=False,
            message="Returned state failed requested bound feasibility tolerance",
        )
    info.setdefault(
        "history_kind",
        "linear_endpoints" if info.get("solver") == "linear" else "native_observations",
    )
    timings = {key: value for key, value in info.items() if key.endswith("_seconds")}
    return SurfaceFitResult(
        geometry,
        np.array(parameters, copy=True),
        bool(info["success"]),
        str(info["message"]),
        int(info.get("iterations", 0)),
        value,
        rms,
        history,
        timings,
        info,
    )


def fit_surface(
    points,
    initial_surface,
    *,
    parameters=None,
    correspondence="fixed",
    initial_parameters=None,
    optimize_weights=False,
    control_mask=None,
    weight_mask=None,
    control_bounds=None,
    weight_bounds=None,
    control_regularization=0.0,
    weight_regularization=0.0,
    objective_scale=1.0,
    tolerance=1e-8,
    max_iterations=1000,
    max_cycles=100,
    max_passes=8,
    record_history=False,
    print_convergence=False,
    callback=None,
    solver_options=None,
):
    """Fit a supplied NURBS patch to positions shaped ``(N, dimension)``.

    ``parameters`` supplies fixed paired labels shaped ``(N, 2)``. For
    ``correspondence='joint'`` or ``'alternating'``, supply optional
    ``initial_parameters`` guesses instead; otherwise the existing surface
    projector initializes them. Degrees, knots and control counts remain fixed.
    Joint parameter optimization requires positive degrees and at least C1
    interior continuity. Alternating fitting retains the projector's one-sided
    treatment of C0 corners.

    Fixed parameters and fixed weights use linear least squares when no free
    coordinate has finite bounds. Other surface optimizations use optional JAX
    reverse AD and exact sparse-Hessian Pounce interior point. The host solve
    itself is not a JAX-transformable operation. NumPy inputs do not require JAX
    for the linear route. Returned geometry uses the input backend.

    Masks broadcast to the corresponding control/weight array; True means free.
    Bounds are lower/upper pairs broadcast to those arrays. If all weights are
    free, initial weights are normalized by W00 and that weight is fixed to one;
    bounds then apply to normalized weights. A pre-existing fixed weight already
    removes the common scale ambiguity, so its original scale is retained.

    The objective is ``objective_scale`` times half the mean squared 3D (or
    input-dimensional) error, plus ``sum(control_regularization * (P-P0)**2)/(2K)``
    and ``weight_regularization * sum((log(w)-log(w0))**2)/(2*nfree_weights)``.
    Regularization defaults to zero; geometric priors and bounds are explicit.
    Iteration/cycle budget exits preserve unsuccessful status.
    """
    begun = perf_counter()
    if correspondence not in ("fixed", "joint", "alternating"):
        raise ValueError("correspondence must be 'fixed', 'joint' or 'alternating'")
    _positive(tolerance, "tolerance")
    _positive(objective_scale, "objective_scale")
    for value, name in (
        (max_iterations, "max_iterations"),
        (max_cycles, "max_cycles"),
        (max_passes, "max_passes"),
    ):
        _budget(value, name)
    if not isinstance(record_history, (bool, np.bool_)) or not isinstance(
        print_convergence, (bool, np.bool_)
    ):
        raise ValueError("record_history and print_convergence must be booleans")
    if callback is not None and not callable(callback):
        raise TypeError("callback must be callable")
    if solver_options is not None and not isinstance(solver_options, dict):
        raise TypeError("solver_options must be a dictionary of native Pounce options")
    if (
        not np.isscalar(weight_regularization)
        or not np.isfinite(weight_regularization)
        or weight_regularization < 0
    ):
        raise ValueError("weight_regularization must be finite and nonnegative")
    layout = FitLayout(
        initial_surface,
        optimize_weights=optimize_weights,
        control_mask=control_mask,
        weight_mask=weight_mask,
        control_bounds=control_bounds,
        weight_bounds=weight_bounds,
    )
    points = validate_points(points, layout.ndim)
    control_penalty(control_regularization, layout)
    if correspondence == "fixed":
        if parameters is None:
            raise ValueError("Fixed correspondence fitting requires parameters")
        if initial_parameters is not None:
            raise ValueError(
                "Use parameters, not initial_parameters, for fixed correspondences"
            )
        uv = validate_parameters(parameters, initial_surface, len(points))
    else:
        if parameters is not None:
            raise ValueError("Use initial_parameters for unknown correspondences")
        uv = (
            None
            if initial_parameters is None
            else validate_parameters(
                initial_parameters, initial_surface, len(points), "initial_parameters"
            )
        )
    options = dict(
        objective_scale=objective_scale,
        control_regularization=control_regularization,
        weight_regularization=weight_regularization,
    )
    preparation = perf_counter() - begun
    bounded = np.isfinite(layout.lower).any() or np.isfinite(layout.upper).any()
    if correspondence == "fixed" and not layout.nweight_variables and not bounded:
        x, info = solve_linear(
            layout,
            points,
            uv,
            control_regularization=control_regularization,
            tolerance=min(tolerance, 1e-10),
            max_iterations=max_iterations,
            print_convergence=print_convergence,
        )
        info["correspondence"] = "fixed"
        info["tolerance"] = tolerance
        frames = _endpoints(layout, x, uv, info["solve_seconds"], phase="surface_fit")
        if callback is not None:
            for frame in frames:
                callback(deepcopy(frame))
        return _finish(
            layout,
            points,
            x,
            uv,
            info,
            frames if record_history else [],
            options,
            preparation,
        )
    begun = perf_counter()
    try:
        import jax
    except ImportError as error:
        raise ImportError(
            "Nonlinear fitting requires NurbsPy's jax extra and pysolver-view[pounce]"
        ) from error
    preparation += perf_counter() - begun
    initialization_compile = initialization_warmup = initialization_projection = 0.0
    with jax.enable_x64():
        projector = None
        if correspondence != "fixed" and (uv is None or not layout.size):
            from ._projection import SurfaceProjector

            begun = perf_counter()
            projector = SurfaceProjector(
                layout.surface(layout.x0),
                points,
                initial_parameters=uv,
                tolerance=min(tolerance * 0.1, 1e-9),
            )
            projection_setup = perf_counter() - begun
            initial_uv = uv
            projected, distances, projection_info = projector.project(
                layout.surface(layout.x0), uv
            )
            if not layout.size:
                uv = projected if initial_uv is None else initial_uv
                frames = _endpoints(
                    layout,
                    layout.x0,
                    uv,
                    projection_info["solve_seconds"],
                    phase="point_projection",
                )
                frames[-1]["parameters"] = projected
                info = dict(
                    success=True,
                    message="Existing NurbsPy point projection converged; surface is fixed",
                    iterations=0,
                    solver="NurbsPy JAX prepared surface projection",
                    nfev=0,
                    njev=0,
                    nhev=0,
                    setup_seconds=projection_setup,
                    compile_seconds=projector.compile_seconds,
                    warmup_seconds=projector.warmup_seconds,
                    solve_seconds=projection_info["solve_seconds"],
                    projection_seconds=projection_info["solve_seconds"],
                    projection_calls=1,
                    correspondence=correspondence,
                    tolerance=tolerance,
                    history_kind="projection_endpoints",
                )
                if print_convergence:
                    print(
                        f"Fixed surface: library projection RMS {np.sqrt(np.mean(distances**2)):.6e}",
                        flush=True,
                    )
                if callback is not None:
                    for frame in frames:
                        callback(deepcopy(frame))
                return _finish(
                    layout,
                    points,
                    layout.x0,
                    projected,
                    info,
                    frames if record_history else [],
                    options,
                    preparation,
                )
            uv = projected
            preparation += projection_setup + projection_info["solve_seconds"]
            initialization_compile = projector.compile_seconds
            initialization_warmup = projector.warmup_seconds
            initialization_projection = projection_info["solve_seconds"]
        begun = perf_counter()
        try:
            from ._nonlinear import SurfaceFitProblem, solve

            import_module("pounce")
        except ImportError as error:
            raise ImportError(
                "Nonlinear fitting requires pip install 'nurbspy[jax]' 'pysolver-view[pounce]>=0.8.2'"
            ) from error
        preparation += perf_counter() - begun
        common = dict(
            tolerance=tolerance,
            max_iterations=max_iterations,
            record_history=record_history,
            print_convergence=print_convergence,
            callback=callback,
            solver_options=solver_options,
        )
        if correspondence == "fixed":
            problem = SurfaceFitProblem(
                layout, points, uv, use_tensor_grid=True, **options
            )
            x, info, frames = solve(problem, **common)
            for frame in frames:
                frame.update(parameters=uv.copy(), phase="surface_fit")
            info["correspondence"] = "fixed"
        elif correspondence == "joint":
            x, uv, info, frames = _joint_fit(
                layout, points, uv, options, common, max_passes
            )
        else:
            x, uv, info, frames = _alternating_fit(
                layout, points, uv, options, common, max_cycles, projector
            )
        info["compile_seconds"] += initialization_compile
        info["warmup_seconds"] += initialization_warmup
        info["initial_projection_seconds"] = initialization_projection
        info["tolerance"] = tolerance
        return _finish(layout, points, x, uv, info, frames, options, preparation)


def _joint_fit(layout, points, uv, options, common, max_passes):
    from ._joint import JointSurfaceFitProblem
    from ._nonlinear import solve

    totals, frames, native = {}, [], []
    start, offset, success = None, 0, False
    user_callback = common["callback"]
    for stage in range(1, max_passes + 1):
        time_offset = totals.get("solve_seconds", 0.0)
        if common["print_convergence"]:
            print(f"Joint surface/parameter pass {stage}", flush=True)
        problem = JointSurfaceFitProblem(layout, points, uv, start=start, **options)

        def observe(row):
            row = deepcopy(row)
            row.update(
                iteration=row["iteration"] + offset,
                seconds=row["seconds"] + time_offset,
                local_pass=stage,
                phase="joint_fit",
            )
            user_callback(row)

        settings = dict(
            common,
            tolerance=min(common["tolerance"] * 0.1, 1e-9),
            callback=observe if user_callback is not None else None,
        )
        full, info, observations = solve(problem, **settings)
        uv = problem.parameters(full)
        begun = perf_counter()
        gradient = problem.gradient(full)[0][layout.size :]
        lower, upper = problem.get_bounds()
        global_lower, global_upper = problem.global_bounds
        multipliers = problem.bound_multipliers
        real_lower = lower[layout.size :] == global_lower[layout.size :]
        real_upper = upper[layout.size :] == global_upper[layout.size :]
        residual = (
            gradient
            - multipliers["lower"][layout.size :] * real_lower
            + multipliers["upper"][layout.size :] * real_upper
        )
        optimality = float(np.max(abs(residual), initial=0.0))
        info["audit_seconds"] = perf_counter() - begun
        for frame in observations:
            frame.update(
                iteration=frame["iteration"] + offset,
                seconds=frame["seconds"] + time_offset,
                local_pass=stage,
                phase="joint_fit",
                parameters=problem.parameters(frame["x"]),
            )
            if frame["event_kind"] == "solution":
                frame["event_kind"] = "local_solution"
            elif stage > 1 and frame["event_kind"] == "initial":
                frame["event_kind"] = "sparsity_restart"
            frames.append(frame)
        for row in info.get("native_history", []):
            row.update(
                iteration=row["iteration"] + offset,
                seconds=row["seconds"] + time_offset,
                local_pass=stage,
                phase="joint_fit",
            )
            native.append(row)
        _sum_info(totals, info)
        offset += info["iterations"]
        success = info["success"] and optimality <= common["tolerance"]
        if common["print_convergence"]:
            print(
                f"  Global parameter stationarity {optimality:.3e}; converged={success}",
                flush=True,
            )
        if success or not info["success"]:
            break
        start = full.copy()
    if frames:
        frames[-1]["event_kind"] = "solution"
    info.update(
        totals,
        success=success,
        correspondence="joint",
        local_passes=stage,
        global_parameter_optimality=optimality,
        native_history=native,
        message=(
            "Native IPM and global parameter stationarity converged"
            if success
            else "Joint fitting stopped before global parameter convergence"
        ),
    )
    return full[: layout.size], uv, info, frames


def _alternating_fit(layout, points, uv, options, common, max_cycles, projector):
    from ._nonlinear import SurfaceFitProblem, solve
    from ._projection import SurfaceProjector

    totals, frames, native = {}, [], []
    offset, success, projection_seconds = 0, False, 0.0
    projection_setup = projection_compile = projection_warmup = 0.0
    if projector is None:
        begun = perf_counter()
        projector = SurfaceProjector(
            layout.surface(layout.x0),
            points,
            initial_parameters=uv,
            tolerance=min(common["tolerance"] * 0.1, 1e-9),
        )
        projection_setup = perf_counter() - begun
        projection_compile, projection_warmup = (
            projector.compile_seconds,
            projector.warmup_seconds,
        )
    problem = SurfaceFitProblem(layout, points, uv, **options)
    user_callback = common["callback"]
    for cycle in range(1, max_cycles + 1):
        time_offset = totals.get("solve_seconds", 0.0) + projection_seconds
        if common["print_convergence"]:
            print(f"Alternating cycle {cycle}: surface fitting", flush=True)

        def observe(row):
            row = deepcopy(row)
            row.update(
                iteration=row["iteration"] + offset,
                seconds=row["seconds"] + time_offset,
                outer_iteration=cycle,
                phase="surface_fit",
            )
            user_callback(row)

        x, info, observations = solve(
            problem,
            **dict(common, callback=observe if user_callback is not None else None),
        )
        for frame in observations:
            frame.update(
                iteration=frame["iteration"] + offset,
                seconds=frame["seconds"] + time_offset,
                outer_iteration=cycle,
                phase="surface_fit",
                parameters=uv.copy(),
            )
            if frame["event_kind"] == "solution":
                frame["event_kind"] = "surface_solution"
            elif cycle > 1 and frame["event_kind"] == "initial":
                frame["event_kind"] = "alternating_restart"
            frames.append(frame)
        for row in info.get("native_history", []):
            row.update(
                iteration=row["iteration"] + offset,
                seconds=row["seconds"] + time_offset,
                outer_iteration=cycle,
                phase="surface_fit",
            )
            native.append(row)
        _sum_info(totals, info)
        offset += info["iterations"]
        projected, distances, projection_info = projector.project(layout.surface(x), uv)
        projection_seconds += projection_info["solve_seconds"]
        begun = perf_counter()
        problem.set_parameters(projected)
        # This correspondence update is part of the explicit block audit below;
        # avoid charging the same basis preparation again to the next solve.
        problem._pending_setup_seconds = 0.0
        multipliers = problem.bound_multipliers
        gradient = problem.gradient(x)[0]
        surface_opt = float(
            np.max(
                abs(gradient - multipliers["lower"] + multipliers["upper"]), initial=0.0
            )
        )
        model = layout.surface(x)
        error = evaluate(layout.initial, *layout.unpack(x), projected) - points
        su = np.asarray(model.get_derivative(*projected.T, 1, 0)).T
        sv = np.asarray(model.get_derivative(*projected.T, 0, 1)).T
        gp = (
            options["objective_scale"]
            / len(points)
            * np.column_stack((np.sum(su * error, axis=1), np.sum(sv * error, axis=1)))
        )
        lower, upper = parameter_domain(layout.initial)
        projected_gradient = projected - np.clip(projected - gp, lower, upper)
        # The checked library projector verifies both one-sided conditions at
        # exact C0 knots. A smooth gradient need not vanish on that corner.
        for axis, (knots, degree) in enumerate(
            ((model.U, model.p), (model.V, model.q))
        ):
            values, counts = np.unique(np.asarray(knots), return_counts=True)
            corners = values[
                (counts >= degree) & (values > lower[axis]) & (values < upper[axis])
            ]
            projected_gradient[np.isin(projected[:, axis], corners), axis] = 0.0
        parameter_opt = float(np.max(abs(projected_gradient), initial=0.0))
        totals["audit_seconds"] = (
            totals.get("audit_seconds", 0.0) + perf_counter() - begun
        )
        if common["record_history"]:
            frames.append(
                dict(
                    iteration=offset,
                    x=x.copy(),
                    parameters=projected.copy(),
                    outer_iteration=cycle,
                    phase="point_projection",
                    event_kind="projection",
                    accepted=True,
                    seconds=totals["solve_seconds"] + projection_seconds,
                )
            )
        if user_callback is not None:
            user_callback(
                dict(
                    iteration=offset,
                    outer_iteration=cycle,
                    phase="point_projection",
                    event_kind="projection",
                    surface_block_optimality=surface_opt,
                    parameter_block_optimality=parameter_opt,
                )
            )
        uv = projected
        success = (
            info["success"]
            and surface_opt <= common["tolerance"]
            and parameter_opt <= common["tolerance"]
        )
        if common["print_convergence"]:
            print(
                f"  Projection RMS {np.sqrt(np.mean(distances**2)):.6e}; surface {surface_opt:.3e}; parameters {parameter_opt:.3e}; converged={success}",
                flush=True,
            )
        if success or not info["success"]:
            break
        problem.x0 = x.copy()
        problem._clear_cache()
    if frames:
        frames[-1]["event_kind"] = "solution"
    message = (
        "Surface and library projection blocks converged"
        if success
        else (
            f"Surface fitting stopped: {info['message']}"
            if not info["success"]
            else "Alternating cycle budget reached before both blocks converged"
        )
    )
    info.update(
        totals,
        success=success,
        correspondence="alternating",
        outer_iterations=cycle,
        message=message,
        native_history=native,
        projection_seconds=projection_seconds,
        fit_seconds=totals["solve_seconds"],
        projection_calls=cycle,
        surface_block_optimality=surface_opt,
        parameter_block_optimality=parameter_opt,
    )
    info["solve_seconds"] += projection_seconds
    info["setup_seconds"] += projection_setup
    info["compile_seconds"] += projection_compile
    info["warmup_seconds"] += projection_warmup
    return x, uv, info, frames
