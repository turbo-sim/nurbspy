"""Optional JAX kernels and the Pounce driver for surface least squares.

This module is imported only for nonlinear fits. The scalar loss is differentiated
in reverse mode; structural coloring recovers its exact sparse Hessian without
forming a dense Hessian or the complete residual Jacobian.
"""

from copy import deepcopy
from time import perf_counter

import numpy as np
import pysolver_view as psv


def hessian_pattern(layout):
    """Lower-triangle least-squares structure for the selected free variables.

    Control coordinates of different spatial components do not couple in this
    loss. Arbitrary user constraints may require a larger Lagrangian pattern.
    """
    nu, nv = layout.shape
    p, q = int(layout.initial.p), int(layout.initial.q)
    dimension, count = layout.ndim, layout.count
    ids = np.arange(count)
    u, v = ids // nv, ids % nv
    pairs = []
    for du in range(-p, p + 1):
        for dv in range(-q, q + 1):
            valid = (u + du >= 0) & (u + du < nu) & (v + dv >= 0) & (v + dv < nv)
            pairs.append((ids[valid], (u[valid] + du) * nv + v[valid] + dv))
    i = np.concatenate([a for a, _ in pairs])
    j = np.concatenate([b for _, b in pairs])
    lower = i >= j
    rows = (dimension * i[lower, None] + np.arange(dimension)).ravel()
    cols = (dimension * j[lower, None] + np.arange(dimension)).ravel()
    rows = np.r_[rows, np.repeat(dimension * count + j, dimension)]
    cols = np.r_[cols, (dimension * i[:, None] + np.arange(dimension)).ravel()]
    rows = np.r_[rows, dimension * count + i[lower]]
    cols = np.r_[cols, dimension * count + j[lower]]

    selected = np.r_[layout.control_indices, dimension * count + layout.weight_indices]
    mapping = np.full((dimension + 1) * count, -1, dtype=int)
    mapping[selected] = np.arange(layout.size)
    keep = (mapping[rows] >= 0) & (mapping[cols] >= 0)
    rows, cols = mapping[rows[keep]], mapping[cols[keep]]
    rows, cols = np.maximum(rows, cols), np.minimum(rows, cols)
    order = np.lexsort((cols, rows))
    rows, cols = rows[order], cols[order]

    control = np.r_[layout.control_indices // dimension, layout.weight_indices]
    kind = np.r_[
        layout.control_indices % dimension,
        np.full(len(layout.weight_indices), dimension),
    ]
    pu, pv = 2 * p + 1, 2 * q + 1
    colors = kind * pu * pv + (control // nv % pu) * pv + control % nv % pv
    _, colors = np.unique(colors, return_inverse=True)
    if len(colors):
        full_rows = np.r_[rows, cols[rows != cols]]
        full_cols = np.r_[cols, rows[rows != cols]]
        keys = full_rows * (int(colors.max()) + 1) + colors[full_cols]
        if np.unique(keys).size != keys.size:
            raise RuntimeError("Surface-fit Hessian coloring has colliding supports")
    return rows, cols, colors


class SurfaceFitProblem(psv.OptimizationProblem):
    """Rational least squares at fixed paired parameters, including variable masks."""

    def __init__(
        self,
        layout,
        points,
        parameters,
        *,
        objective_scale=1.0,
        control_regularization=0.0,
        weight_regularization=0.0,
        start=None,
        use_tensor_grid=False,
    ):
        begun = perf_counter()
        import jax
        import jax.numpy as jnp
        from ._common import basis_table

        self._configure(
            layout,
            points,
            objective_scale,
            control_regularization,
            weight_regularization,
        )
        self.x0 = np.array(
            layout.x0 if start is None else start, dtype=float, copy=True
        )
        if self.x0.shape != (layout.size,):
            raise ValueError("The fitting start has the wrong number of variables")
        self._parameters = np.array(parameters, dtype=float, copy=True)
        self.use_tensor_grid = False
        if use_tensor_grid:
            u, iu = np.unique(self._parameters[:, 0], return_inverse=True)
            v, iv = np.unique(self._parameters[:, 1], return_inverse=True)
            self.use_tensor_grid = len(u) * len(v) == len(parameters) and len(
                np.unique(iu * len(v) + iv)
            ) == len(parameters)
        if self.use_tensor_grid:
            from scipy.interpolate import BSpline

            a = jnp.asarray(
                BSpline.design_matrix(
                    u, np.asarray(layout.initial.U), int(layout.initial.p)
                ).toarray()
            )
            b = jnp.asarray(
                BSpline.design_matrix(
                    v, np.asarray(layout.initial.V), int(layout.initial.q)
                ).toarray()
            )
            target = np.empty((len(u), len(v), layout.ndim))
            target[iu, iv] = points
            target = jnp.asarray(target)
            self._pair_data = ()

            def objective(x):
                controls, weights = self._unpack_jax(x)
                net = controls.reshape(*layout.shape, layout.ndim)
                weights = weights.reshape(layout.shape)
                denominator = a @ weights @ b.T
                numerator = jnp.einsum(
                    "ui,ijc,vj->uvc", a, net * weights[..., None], b, optimize="optimal"
                )
                error = numerator / denominator[..., None] - target
                return objective_scale * 0.5 * jnp.mean(
                    jnp.sum(error**2, axis=-1)
                ) + self._regularization(x, controls)

        else:
            indices, values = basis_table(layout.initial, self._parameters)
            self._pair_data = jnp.asarray(indices), jnp.asarray(values)

            def objective(x, indices, basis):
                controls, weights = self._unpack_jax(x)
                weighted = basis * weights[indices]
                prediction = jnp.sum(weighted[..., None] * controls[indices], axis=1)
                prediction /= jnp.sum(weighted, axis=1)[:, None]
                error = prediction - self._target
                return objective_scale * 0.5 * jnp.mean(
                    jnp.sum(error**2, axis=1)
                ) + self._regularization(x, controls)

        self.bounds = layout.lower.copy(), layout.upper.copy()
        self.rows, self.columns, colors = hessian_pattern(layout)
        self.color_count = int(colors.max()) + 1 if len(colors) else 0
        seeds = jnp.asarray(np.eye(self.color_count)[colors].T)
        decode, rows = jnp.asarray(colors[self.columns]), jnp.asarray(self.rows)

        gradient = jax.grad(objective)

        def sparse_hessian(x, *data):
            if not self.color_count:
                return jnp.empty((0,), dtype=x.dtype)
            _, product = jax.linearize(lambda point: gradient(point, *data), x)
            compressed = (
                jax.vmap(product)(seeds)
                if self.use_tensor_grid
                else jax.lax.map(product, seeds, batch_size=4)
            )
            return compressed[decode, rows]

        kernels = [objective, jax.value_and_grad(objective), sparse_hessian]
        compiled = self._compile(kernels, (jnp.asarray(self.x0), *self._pair_data))
        self._value = lambda x: compiled[0](x, *self._pair_data)
        self._value_gradient = lambda x: compiled[1](x, *self._pair_data)
        self._hessian = lambda x: compiled[2](x, *self._pair_data)
        self._finish_setup(begun)

    def _configure(
        self,
        layout,
        points,
        objective_scale,
        control_regularization,
        weight_regularization,
    ):
        """Shared geometry layout and scale-independent regularization kernels."""
        import jax
        import jax.numpy as jnp

        if not jax.config.x64_enabled:
            raise RuntimeError(
                "Construct and solve fitting problems inside jax.enable_x64()"
            )
        self.layout, self.initial = layout, layout.initial
        self.surface_size = layout.size
        self.count, self.ndim = layout.count, layout.ndim
        self.points = np.array(points, dtype=float, copy=True)
        self.objective_scale = float(objective_scale)
        if not np.isfinite(objective_scale) or objective_scale <= 0:
            raise ValueError("objective_scale must be finite and positive")
        reg = np.asarray(control_regularization, dtype=float)
        if reg.shape != (layout.count, layout.ndim):
            reg = (
                np.broadcast_to(reg, np.shape(layout.initial.P))
                .reshape(layout.ndim, -1)
                .T
            )
        if not np.all(np.isfinite(reg)) or np.any(reg < 0):
            raise ValueError("control_regularization must be finite and nonnegative")
        if (
            not np.isscalar(weight_regularization)
            or not np.isfinite(weight_regularization)
            or weight_regularization < 0
        ):
            raise ValueError("weight_regularization must be finite and nonnegative")
        self._target = jnp.asarray(points)
        reference = jnp.asarray(layout.controls0)
        weights0 = jnp.asarray(layout.weights0)
        logweights0 = jnp.log(weights0)
        controls_index = jnp.asarray(layout.control_indices)
        weight_index = jnp.asarray(layout.weight_indices)
        nc = len(layout.control_indices)
        reg = jnp.asarray(reg)

        def unpack(x):
            controls = (
                reference.ravel()
                .at[controls_index]
                .set(x[:nc])
                .reshape(reference.shape)
            )
            weights = weights0.at[weight_index].set(jnp.exp(x[nc : layout.size]))
            return controls, weights

        def logweights(x):
            return logweights0.at[weight_index].set(x[nc : layout.size])

        def regularization(x, controls):
            value = 0.5 / layout.count * jnp.sum(reg * (controls - reference) ** 2)
            value += (
                0.5
                * weight_regularization
                / max(len(layout.weight_indices), 1)
                * jnp.sum((x[nc : layout.size] - logweights0[weight_index]) ** 2)
            )
            return objective_scale * value

        self._unpack_jax, self._logweights_jax = unpack, logweights
        self._regularization = regularization
        self.native_options = {}
        self._clear_cache()

    def _compile(self, kernels, arguments):
        import jax

        self.compile_seconds = self.warmup_seconds = 0.0
        compiled = []
        for kernel in kernels:
            begun = perf_counter()
            function = jax.jit(kernel).lower(*arguments).compile()
            self.compile_seconds += perf_counter() - begun
            begun = perf_counter()
            jax.block_until_ready(function(*arguments))
            self.warmup_seconds += perf_counter() - begun
            compiled.append(function)
        return compiled

    def _finish_setup(self, begun):
        self.setup_seconds = perf_counter() - begun
        self._pending_setup_seconds = self.setup_seconds
        self._pending_compile_seconds = self.compile_seconds
        self._pending_warmup_seconds = self.warmup_seconds
        self.value_seconds = self.gradient_seconds = self.hessian_seconds = 0.0

    def _clear_cache(self):
        self._x = self._f = self._g = None

    def set_parameters(self, parameters):
        """Change correspondences while retaining the compiled fixed-fit kernels."""
        if self.use_tensor_grid:
            raise ValueError(
                "Construct a new tensor-grid problem when its fixed labels change"
            )
        from ._common import basis_table
        import jax.numpy as jnp

        begun = perf_counter()
        parameters = np.asarray(parameters, dtype=float)
        if parameters.shape != self._parameters.shape:
            raise ValueError("Updated parameters must preserve the observation count")
        indices, values = basis_table(self.initial, parameters)
        self._pair_data = jnp.asarray(indices), jnp.asarray(values)
        self._parameters = parameters.copy()
        self._clear_cache()
        self._pending_setup_seconds += perf_counter() - begun

    def parameters(self, x):
        return self._parameters.copy()

    def surface(self, x):
        return self.layout.surface(np.asarray(x)[: self.surface_size])

    def fitness(self, x):
        begun = perf_counter()
        if self._x is None or not np.array_equal(x, self._x):
            self._f = float(self._value(x))
            self._x, self._g = np.array(x, copy=True), None
        self.value_seconds += perf_counter() - begun
        return np.array([self._f])

    def gradient(self, x):
        begun = perf_counter()
        if self._g is None or not np.array_equal(x, self._x):
            value, gradient = self._value_gradient(x)
            self._f, self._g, self._x = (
                float(value),
                np.asarray(gradient),
                np.array(x, copy=True),
            )
        self.gradient_seconds += perf_counter() - begun
        return self._g.copy()[None]

    def hessians(self, x):
        begun = perf_counter()
        result = np.asarray(self._hessian(x))
        self.hessian_seconds += perf_counter() - begun
        return result[None]

    def hessian_sparsity(self):
        return self.rows, self.columns

    def get_bounds(self):
        return self.bounds

    def get_nec(self):
        return 0

    def get_nic(self):
        return 0


def solve(
    problem,
    *,
    tolerance=1e-8,
    max_iterations=1000,
    record_history=False,
    print_convergence=False,
    callback=None,
    solver_options=None,
):
    """Run exact-Hessian Pounce IPM, returning its authoritative point and status.

    Coordinates exposed by PySolverView's native callback are last objective
    evaluations with unknown acceptance. They are labeled accordingly; the
    initial point and returned solution are separate, authoritative snapshots.
    """
    begun = perf_counter()
    import pounce  # Load the native backend before starting the warm solve clock.

    options = dict(solver_options or {})
    forbidden = {"tol", "max_iter", "algorithm"}.intersection(options)
    if forbidden:
        raise ValueError(
            "Use fit_surface tolerance/max_iterations rather than native overrides: "
            + ", ".join(sorted(forbidden))
        )
    if options.get("hessian_approximation", "exact") != "exact":
        raise ValueError("Surface fitting requires hessian_approximation='exact'")
    if options.get("stop_on_kkt", False):
        raise ValueError("Separate KKT checks are disabled for surface fitting")
    options = {
        "hessian_approximation": "exact",
        "mu_strategy": "adaptive",
        "nlp_scaling_method": "none",
        "bound_relax_factor": 0.0,
        "stop_on_kkt": False,
        "bound_push": 1e-9,
        "bound_frac": 1e-9,
        **getattr(problem, "native_options", {}),
        **options,
    }
    raw = []
    last_native = {}
    recording_seconds = 0.0
    problem.value_seconds = problem.gradient_seconds = problem.hessian_seconds = 0.0
    if print_convergence:
        print(
            " Iteration       Objective      Primal inf        Dual inf        Barrier     Seconds",
            flush=True,
        )

    def observe(entry):
        nonlocal last_native, recording_seconds
        started = perf_counter()
        last_native = dict(entry["native_metrics"])
        row = {
            key: value
            for key, value in entry.items()
            if key not in ("x", "native_x", "fitness")
        }
        row.update(
            event_kind="native_observation",
            accepted=None,
            coordinate_source="last_objective_evaluation",
            seconds=started - solve_started,
            objective=float(last_native.get("obj_value", np.nan)),
            objective_source="native_iteration_metrics",
        )
        if record_history or callback is not None:
            row["x"] = (
                np.array(entry["x"], copy=True) if entry.get("x") is not None else None
            )
        if record_history:
            raw.append(row)
        if print_convergence:
            print(
                f" {int(entry['iteration']):9d} {last_native.get('obj_value', np.nan):15.7e}"
                f" {last_native.get('inf_pr', np.nan):15.5e} {last_native.get('inf_du', np.nan):15.5e}"
                f" {last_native.get('mu', np.nan):14.4e} {row['seconds']:11.3f}",
                flush=True,
            )
        if callback is not None:
            callback(deepcopy(row))
        recording_seconds += perf_counter() - started

    solver = psv.OptimizationSolver(
        problem,
        library="pounce",
        method="ipm",
        tolerance=tolerance,
        max_iterations=max_iterations,
        check_kkt=False,
        print_convergence=False,
        update_on="iteration",
        backend_callback=observe,
        extra_options=options,
    )
    setup_seconds = problem._pending_setup_seconds + perf_counter() - begun
    compile_seconds = problem._pending_compile_seconds
    warmup_seconds = problem._pending_warmup_seconds
    problem._pending_setup_seconds = problem._pending_compile_seconds = (
        problem._pending_warmup_seconds
    ) = 0.0
    solve_started = perf_counter()
    final = np.asarray(solver.solve(problem.x0))
    elapsed = perf_counter() - solve_started
    result = solver.result
    problem.bound_multipliers = solver.backend_result.native_multipliers
    model_times = dict(
        model_value_seconds=problem.value_seconds,
        model_gradient_seconds=problem.gradient_seconds,
        model_hessian_seconds=problem.hessian_seconds,
    )
    info = dict(
        success=bool(result.success),
        message=str(result.message),
        status=int(result.status),
        solver="Pounce IPM / exact sparse JAX Hessian",
        variables=len(final),
        iterations=int(result.nit),
        nfev=int(result.nfev),
        njev=int(result.njev),
        nhev=int(result.nhev),
        setup_seconds=setup_seconds,
        compile_seconds=compile_seconds,
        warmup_seconds=warmup_seconds,
        solve_seconds=elapsed,
        recording_seconds=recording_seconds,
        kkt_seconds=float(result.kkt_time),
        hessian_seconds=float(result.hessian_time),
        hessian_nnz=len(problem.rows),
        hessian_colors=problem.color_count,
        native_dual_infeasibility=last_native.get("inf_du"),
        native_tolerance=tolerance,
        bound_multipliers=problem.bound_multipliers,
        native_history=[
            {key: value for key, value in row.items() if key != "x"} for row in raw
        ],
        **model_times,
    )
    history = []
    if record_history:
        history = [
            dict(
                iteration=0,
                x=problem.x0.copy(),
                seconds=0.0,
                event_kind="initial",
                accepted=True,
            )
        ]
        history += [
            row for row in raw if row.get("x") is not None and row["iteration"] > 0
        ]
        history += [
            dict(
                iteration=int(result.nit),
                x=final.copy(),
                seconds=elapsed,
                event_kind="solution",
                accepted=True,
            )
        ]
    if print_convergence:
        print(
            f" {result.message}; {result.nit} iterations, warm solve {elapsed:.3f} s",
            flush=True,
        )
    return final, info, history
