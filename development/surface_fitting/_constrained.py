"""Case 05: native Pounce IPM with sparse Jacobian and exact Lagrangian Hessian.

The native interface combines constraint Hessians with their multipliers inside
AD. It avoids storing one Hessian per orientation constraint or a dense Jacobian.
"""

from time import perf_counter
import numpy as np
from _advanced import constrained_problem, area_quadrature, paired_basis, terms
from _nonlinear import hessian_pattern, OBJECTIVE_SCALE
from _experiment import evaluate


class ConstrainedSurfaceProblem:
    """Bounded heights/interior weights with area and sampled orientation limits."""

    def __init__(self, initial, observations):
        """Compile the exact weighted Hessian and declare fixed sparse structures."""
        import jax
        import jax.numpy as jnp

        self.initial, self.observations = initial, observations
        self.x0, self.bounds, self.surface, self.values, self.settings = (
            constrained_problem(initial, observations)
        )
        count, dimension = initial.W.size, len(self.x0)
        interior = np.ones(initial.W.shape, dtype=bool)
        interior[[0, -1], :] = False
        interior[:, [0, -1]] = False
        free = np.flatnonzero(interior.ravel())
        mapping = np.full(4 * count - 1, -1, dtype=int)
        mapping[3 * np.arange(count) + 2] = np.arange(count)
        mapping[3 * count + free - 1] = count + np.arange(len(free))
        r, c, _ = hessian_pattern(initial.W.shape, (3, 3))
        valid = (mapping[r] >= 0) & (mapping[c] >= 0)
        self.rows, self.columns = mapping[r[valid]], mapping[c[valid]]
        ids = np.r_[np.arange(count), free]
        kind = np.r_[np.zeros(count, dtype=int), np.ones(len(free), dtype=int)]
        colors = (
            kind * 49
            + (ids // initial.W.shape[1] % 7) * 7
            + ids % initial.W.shape[1] % 7
        )
        _, colors = np.unique(colors, return_inverse=True)
        self.color_count = int(colors.max()) + 1
        full_r = np.r_[self.rows, self.columns[self.rows != self.columns]]
        full_c = np.r_[self.columns, self.rows[self.rows != self.columns]]
        keys = full_r * self.color_count + colors[full_c]
        if np.unique(keys).size != len(keys):
            raise RuntimeError("Constraint Hessian coloring collision")
        state = self.values(self.x0)
        guard_jac = state["gjac"][1:].tocoo()
        self.jrows = np.r_[np.zeros(dimension, dtype=int), guard_jac.row + 1]
        self.jcols = np.r_[np.arange(dimension), guard_jac.col]
        self.constraint_count = len(state["g"])
        axis = observations["axis"]
        u, v = np.meshgrid(axis, axis, indexing="ij")
        training = paired_basis(initial, np.stack((u.ravel(), v.ravel()), axis=1))
        uv, omega = area_quadrature(
            initial, max(6, int(np.ceil(20 / (initial.W.shape[0] - 3))))
        )
        am = paired_basis(initial, uv, derivatives=True)
        axis = np.linspace(0, 1, 2 * initial.W.shape[0] + 1)
        u, v = np.meshgrid(axis, axis, indexing="ij")
        gm = paired_basis(
            initial, np.stack((u.ravel(), v.ravel()), axis=1), derivatives=True
        )
        reference = jnp.asarray(initial.P.reshape(3, -1).T)
        fi = jnp.asarray(free)
        target = jnp.asarray(observations["points"])
        omega = jnp.asarray(omega)

        def arrays(matrices):
            first = matrices[0]
            indices = jnp.asarray(first.indices.reshape(-1, 16))
            return (indices, *[jnp.asarray(m.data.reshape(-1, 16)) for m in matrices])

        td, ad, gd = arrays((training,)), arrays(am), arrays(gm)

        def geometry(x, data):
            indices, b, *derivatives = data
            controls = reference.at[:, 2].set(x[:count])
            weights = jnp.ones(count, dtype=x.dtype).at[fi].set(jnp.exp(x[count:]))
            local = controls[indices]
            weighted = b * weights[indices]
            denominator = jnp.sum(weighted, axis=1)
            s = jnp.sum(weighted[..., None] * local, axis=1) / denominator[:, None]
            if not derivatives:
                return s
            tangents = []
            for derivative in derivatives:
                weighted = derivative * weights[indices]
                tangents.append(
                    (
                        jnp.sum(weighted[..., None] * local, axis=1)
                        - s * jnp.sum(weighted, axis=1)[:, None]
                    )
                    / denominator[:, None]
                )
            return s, *tangents

        def lagrangian(x, multipliers, factor):
            residual = geometry(x, td) - target
            value = (
                factor * OBJECTIVE_SCALE * 0.5 * jnp.mean(jnp.sum(residual**2, axis=1))
            )
            _, su, sv = geometry(x, ad)
            area = omega @ jnp.linalg.norm(jnp.cross(su, sv), axis=1)
            _, su, sv = geometry(x, gd)
            orientation = su[:, 0] * sv[:, 1] - su[:, 1] * sv[:, 0]
            return (
                value
                + multipliers[0] * (area / self.settings["area_limit"] - 1)
                + multipliers[1:] @ (0.05 - orientation)
            )

        gradient = jax.grad(lagrangian, 0)
        seeds = jnp.asarray(np.eye(self.color_count)[colors].T)
        rows, decode = jnp.asarray(self.rows), jnp.asarray(colors[self.columns])

        def sparse(x, multipliers, factor):
            _, product = jax.linearize(
                lambda point: gradient(point, multipliers, factor), x
            )
            compressed = jax.lax.map(product, seeds, batch_size=2)
            return compressed[decode, rows]

        begun = perf_counter()
        arguments = (
            jnp.asarray(self.x0),
            jnp.zeros(self.constraint_count),
            jnp.asarray(1.0),
        )
        self._hessian = jax.jit(sparse).lower(*arguments).compile()
        self.compile_seconds = perf_counter() - begun
        begun = perf_counter()
        self._hessian(*arguments).block_until_ready()
        self.warmup_seconds = perf_counter() - begun
        self.value_seconds = self.gradient_seconds = self.hessian_seconds = 0.0
        self.nfev = self.njev = self.nhev = 0
        self.history = []
        self.last = None

    def objective(self, x):
        """Evaluate the scaled data loss and cache matching observation diagnostics."""
        begun = perf_counter()
        state = self.values(x)
        value = OBJECTIVE_SCALE * 0.5 * (state["residual"] @ state["residual"])
        self.last = dict(
            x=np.array(x, copy=True),
            area=state["area"],
            constraint_violation=float(max(0, state["g"].max())),
        )
        self.value_seconds += perf_counter() - begun
        self.nfev += 1
        return float(value)

    def gradient(self, x):
        """Return the exact analytic scalar gradient from the sparse data Jacobian."""
        begun = perf_counter()
        state = self.values(x)
        result = OBJECTIVE_SCALE * (state["jac"].T @ state["residual"])
        self.gradient_seconds += perf_counter() - begun
        self.njev += 1
        return np.asarray(result)

    def constraints(self, x):
        """Nonpositive area and sampled orientation inequalities."""
        begun = perf_counter()
        result = self.values(x)["g"].copy()
        self.value_seconds += perf_counter() - begun
        return result

    def jacobianstructure(self):
        """Keep area row dense and orientation rows local, with a fixed union pattern."""
        return self.jrows, self.jcols

    def jacobian(self, x):
        """Return sparse constraint derivatives in the declared order."""
        begun = perf_counter()
        result = np.asarray(self.values(x)["gjac"][self.jrows, self.jcols]).ravel()
        self.gradient_seconds += perf_counter() - begun
        return result

    def hessianstructure(self):
        """Sparse lower triangle shared by the objective and constraint Hessians."""
        return self.rows, self.columns

    def hessian(self, x, lagrange, obj_factor):
        """Exact weighted Lagrangian curvature, excluding Pounce's barrier terms."""
        begun = perf_counter()
        result = np.asarray(self._hessian(x, lagrange, np.asarray(obj_factor)))
        self.hessian_seconds += perf_counter() - begun
        self.nhev += 1
        return result

    def intermediate(
        self,
        alg_mod,
        iteration,
        objective,
        inf_pr,
        inf_du,
        mu,
        d_norm,
        regularization,
        alpha_du,
        alpha_pr,
        ls_trials,
    ):
        """Retain native metrics and occasional last-evaluation coordinates truthfully."""
        record = dict(
            iteration=int(iteration),
            seconds=perf_counter() - self.solve_started,
            accepted=None,
            coordinate_source="last_objective_evaluation",
            event_kind="native_observation",
            native_metrics=dict(
                alg_mod=int(alg_mod),
                obj_value=float(objective),
                inf_pr=float(inf_pr),
                inf_du=float(inf_du),
                mu=float(mu),
                regularization_size=float(regularization),
                ls_trials=int(ls_trials),
            ),
        )
        if int(alg_mod) == 0 and self.last is not None and iteration % 5 == 0:
            record.update(self.last)
        self.history.append(record)
        if iteration % 10 == 0:
            print(
                f"  IPM {iteration:4d}: objective {objective:.6g}, primal {inf_pr:.3g}, dual {inf_du:.3g}",
                flush=True,
            )
        return True


def constrained(initial, observations, *, max_iterations=1000, tolerance=1e-8):
    """Solve the constrained fit with native sparse Pounce IPM and finer final audits."""
    begun = perf_counter()
    import jax
    import pounce

    with jax.enable_x64():
        problem = ConstrainedSurfaceProblem(initial, observations)
        lower, upper = problem.bounds
        native = pounce.Problem(
            n=len(lower),
            m=problem.constraint_count,
            problem_obj=problem,
            lb=lower,
            ub=upper,
            cl=np.full(problem.constraint_count, -np.inf),
            cu=np.zeros(problem.constraint_count),
        )
        for key, value in dict(
            algorithm="interior-point",
            hessian_approximation="exact",
            mu_strategy="adaptive",
            tol=tolerance,
            max_iter=max_iterations,
            print_level=0,
            nlp_scaling_method="none",
            bound_relax_factor=0.0,
        ).items():
            native.add_option(key, value)
        setup = perf_counter() - begun
        problem.solve_started = perf_counter()
        final, result = native.solve(problem.x0)
        elapsed = perf_counter() - problem.solve_started
        timing = dict(
            model_value_seconds=problem.value_seconds,
            model_gradient_seconds=problem.gradient_seconds,
            model_hessian_seconds=problem.hessian_seconds,
        )
        snapshots = [dict(x=problem.x0, iteration=0, event_kind="initial", seconds=0.0)]
        snapshots += [r for r in problem.history if "x" in r and r["iteration"] > 0]
        snapshots += [
            dict(
                x=final,
                iteration=int(result["iter_count"]),
                event_kind="solution",
                seconds=elapsed,
            )
        ]
        history = []
        for frame in snapshots:
            model = problem.surface(frame["x"])
            state = problem.values(frame["x"])
            residual = evaluate(model, observations["axis"]) - observations["points"]
            history.append(
                dict(
                    iteration=frame["iteration"],
                    event_kind=frame["event_kind"],
                    seconds=frame["seconds"],
                    surface=model,
                    area=state["area"],
                    constraint_violation=float(max(0, state["g"].max())),
                    training_rms=float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
                )
            )
        fitted = problem.surface(final)
        uv, omega = area_quadrature(
            fitted, max(8, int(np.ceil(24 / (initial.W.shape[0] - 3))))
        )
        b, bu, bv = paired_basis(fitted, uv, derivatives=True)
        values = terms(b, fitted.P.reshape(3, -1).T, fitted.W.ravel(), bu, bv)
        area = float(
            omega @ np.linalg.norm(np.cross(values["su"], values["sv"]), axis=1)
        )
        axis = np.linspace(0, 1, 4 * initial.W.shape[0] + 1)
        u, v = np.meshgrid(axis, axis, indexing="ij")
        b, bu, bv = paired_basis(
            fitted, np.stack((u.ravel(), v.ravel()), axis=1), derivatives=True
        )
        values = terms(b, fitted.P.reshape(3, -1).T, fitted.W.ravel(), bu, bv)
        orientation = float(
            np.min(
                values["su"][:, 0] * values["sv"][:, 1]
                - values["su"][:, 1] * values["sv"][:, 0]
            )
        )
        success = (
            int(result["status"]) == 0
            and area <= problem.settings["area_limit"] * (1 + 2e-5)
            and orientation >= 0.05 - 1e-6
        )
        native_history = [
            {k: v for k, v in r.items() if k != "x"} for r in problem.history
        ]
        info = dict(
            **problem.settings,
            success=success,
            status=int(result["status"]),
            message=str(result.get("status_msg", result["status"])),
            solver="Pounce IPM / exact sparse Lagrangian Hessian and Jacobian",
            iterations=int(result["iter_count"]),
            nfev=problem.nfev,
            njev=problem.njev,
            nhev=problem.nhev,
            solve_seconds=elapsed,
            setup_seconds=setup,
            compile_seconds=problem.compile_seconds,
            warmup_seconds=problem.warmup_seconds,
            **timing,
            kkt_seconds=0.0,
            hessian_nnz=len(problem.rows),
            hessian_colors=problem.color_count,
            tolerance=tolerance,
            checked_area=area,
            checked_minimum_xy_jacobian=orientation,
            native_history=native_history,
            history_kind="native_observations",
            constraint_violation=history[-1]["constraint_violation"],
            native_dual_infeasibility=problem.history[-1]["native_metrics"]["inf_du"],
        )
    return fitted, info, history
