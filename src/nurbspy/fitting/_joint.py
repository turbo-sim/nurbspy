"""Exact sparse joint surface/point-parameter fitting in moving support windows."""

from time import perf_counter

import numpy as np

from ._nonlinear import SurfaceFitProblem, hessian_pattern


def _nonempty_spans(knots, degree, count):
    spans = np.flatnonzero(np.diff(knots) > 0)
    spans = spans[(spans >= degree) & (spans < count)]
    if not len(spans):
        raise ValueError("Joint fitting needs nonempty knot spans")
    return spans


def _candidate_count(knots, degree, count):
    spans = _nonempty_spans(knots, degree, count)
    gap = int(np.max(np.diff(spans), initial=0))
    return min(count, degree + 1 + gap)


def parameter_windows(initial, parameters):
    """Cover each parameter by adjacent knot spans and all possible local controls."""
    boxes, bases, candidates = [], [], []
    for axis, knots, count, degree in zip(
        np.asarray(parameters).T,
        (np.asarray(initial.U), np.asarray(initial.V)),
        np.shape(initial.W),
        (int(initial.p), int(initial.q)),
    ):
        nonempty = _nonempty_spans(knots, degree, count)
        span = np.searchsorted(knots, axis, side="right") - 1
        position = np.clip(
            np.searchsorted(nonempty, span, side="right") - 1, 0, len(nonempty) - 1
        )
        span = nonempty[position]
        fraction = (axis - knots[span]) / (knots[span + 1] - knots[span])
        first_position = np.clip(
            position - (fraction < 0.5), 0, max(0, len(nonempty) - 2)
        )
        first = nonempty[first_position]
        last = nonempty[np.minimum(first_position + 1, len(nonempty) - 1)]
        boxes.append((knots[first], knots[last + 1]))
        bases.append(first - degree)
        # Adjacent nonempty spans may be separated by a repeated knot. Include
        # their full union of control supports rather than degree+2 controls.
        candidate_count = _candidate_count(knots, degree, count)
        candidate = first[:, None] - degree + np.arange(candidate_count)
        candidates.append((candidate, candidate <= last[:, None]))
    (iu, valid_u), (iv, valid_v) = candidates
    nu, nv = np.shape(initial.W)
    valid = (
        (iu[:, :, None] >= 0)
        & (iu[:, :, None] < nu)
        & (iv[:, None, :] >= 0)
        & (iv[:, None, :] < nv)
        & valid_u[:, :, None]
        & valid_v[:, None, :]
    )
    indices = (
        np.clip(iu, 0, nu - 1)[:, :, None] * nv + np.clip(iv, 0, nv - 1)[:, None, :]
    ).reshape(len(parameters), -1)
    lower = np.stack([box[0] for box in boxes], axis=1)
    upper = np.stack([box[1] for box in boxes], axis=1)
    if np.any(lower >= upper):
        raise ValueError("Joint fitting needs nonempty local knot-span windows")
    return (
        lower,
        upper,
        np.stack(bases, axis=1),
        indices,
        valid.reshape(len(parameters), -1),
    )


class JointSurfaceFitProblem(SurfaceFitProblem):
    """One joint IPM pass, with a Hessian valid throughout each local window.

    The caller may recenter windows and restart from the returned full vector.
    These restarts always optimize surface variables and all u/v together.
    Positive degrees and at least C1 continuity at interior knots are required
    for the smooth global stationarity check; repeated C1/C2 knots are supported.
    """

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
    ):
        begun = perf_counter()
        import jax
        import jax.numpy as jnp

        self._configure(
            layout,
            points,
            objective_scale,
            control_regularization,
            weight_regularization,
        )
        self.native_options = {"mu_min": 1e-14}
        count, ndim, size = layout.count, layout.ndim, layout.size
        samples = len(points)
        self.x0 = np.array(
            (
                np.r_[layout.x0, np.asarray(parameters).ravel()]
                if start is None
                else start
            ),
            dtype=float,
            copy=True,
        )
        if self.x0.shape != (size + 2 * samples,):
            raise ValueError(
                "The joint fitting start has the wrong number of variables"
            )
        parameters = self.parameters(self.x0)
        p, q = int(layout.initial.p), int(layout.initial.q)
        if p < 1 or q < 1:
            raise ValueError(
                "Joint fitting requires positive degrees in both parameter directions"
            )
        for knots, degree in (
            (np.asarray(layout.initial.U), p),
            (np.asarray(layout.initial.V), q),
        ):
            values, multiplicity = np.unique(knots, return_counts=True)
            interior = (values > knots[degree]) & (values < knots[-degree - 1])
            if np.any(multiplicity[interior] >= degree):
                raise ValueError(
                    "Joint fitting requires at least C1 continuity across interior knots"
                )
        low, high, bases, indices, valid = parameter_windows(layout.initial, parameters)
        nu, nv = layout.shape
        candidate_u = _candidate_count(np.asarray(layout.initial.U), p, nu)
        candidate_v = _candidate_count(np.asarray(layout.initial.V), q, nv)
        domain_lower = np.array([layout.initial.U[p], layout.initial.V[q]], dtype=float)
        domain_upper = np.array(
            [layout.initial.U[-p - 1], layout.initial.V[-q - 1]], dtype=float
        )
        self.global_bounds = (
            np.r_[layout.lower, np.tile(domain_lower, samples)],
            np.r_[layout.upper, np.tile(domain_upper, samples)],
        )
        lower, upper = (array.copy() for array in self.global_bounds)
        lower[size:], upper[size:] = low.ravel(), high.ravel()
        self.bounds = lower, upper
        last_spans = np.column_stack(
            (
                np.searchsorted(np.asarray(layout.initial.U), high[:, 0], side="left")
                - 1,
                np.searchsorted(np.asarray(layout.initial.V), high[:, 1], side="left")
                - 1,
            )
        )
        ji, jb, je, jvalid = (
            jnp.asarray(indices),
            jnp.asarray(bases),
            jnp.asarray(last_spans),
            jnp.asarray(valid),
        )
        knots_u, knots_v = jnp.asarray(layout.initial.U), jnp.asarray(layout.initial.V)
        local_count = indices.shape[1]
        local_control_size = local_count * ndim
        local_size = local_count * (ndim + 1)

        def axis_basis(value, base, last_span, knots, n, degree, candidate_count):
            ids = base + jnp.arange(candidate_count + degree)
            at = lambda positions: knots[jnp.clip(positions, 0, len(knots) - 1)]
            inside = (value >= at(ids)) & (value < at(ids + 1)) & (ids < n)
            # At an artificial upper endpoint use the polynomial limit from
            # inside this window. Repeated knots can otherwise select controls
            # from the next, excluded span. Values/first derivatives agree for
            # C1 patches, while the in-window Hessian stays exact.
            values = jnp.where(
                value == knots[last_span + 1], ids == last_span, inside
            ).astype(value.dtype)
            for order in range(1, degree + 1):
                ids = base + jnp.arange(candidate_count + degree - order)
                dl, dr = at(ids + order) - at(ids), at(ids + order + 1) - at(ids + 1)
                left = (
                    jnp.where(dl > 0, (value - at(ids)) / jnp.where(dl > 0, dl, 1), 0)
                    * values[:-1]
                )
                right = (
                    jnp.where(
                        dr > 0,
                        (at(ids + order + 1) - value) / jnp.where(dr > 0, dr, 1),
                        0,
                    )
                    * values[1:]
                )
                values = left + right
            return values

        def point_loss(local, parameter, point, base, last_span, is_valid):
            controls = local[:local_control_size].reshape(local_count, ndim)
            weights = jnp.exp(local[local_control_size:])
            bu = axis_basis(
                parameter[0], base[0], last_span[0], knots_u, nu, p, candidate_u
            )
            bv = axis_basis(
                parameter[1], base[1], last_span[1], knots_v, nv, q, candidate_v
            )
            weighted = (bu[:, None] * bv[None, :]).ravel() * weights * is_valid
            prediction = jnp.sum(weighted[:, None] * controls, axis=0) / jnp.sum(
                weighted
            )
            error = prediction - point
            return objective_scale / (2 * samples) * jnp.sum(error**2)

        def local_values(surface_variables):
            controls, _ = self._unpack_jax(surface_variables)
            selected_controls = controls[ji].reshape(samples, local_control_size)
            selected_weights = self._logweights_jax(surface_variables)[ji]
            return jnp.concatenate((selected_controls, selected_weights), axis=1)

        def objective_parts(surface_variables, parameters):
            loss = jax.vmap(point_loss)(
                local_values(surface_variables),
                parameters,
                self._target,
                jb,
                je,
                jvalid,
            ).sum()
            controls, _ = self._unpack_jax(surface_variables)
            return loss + self._regularization(surface_variables, controls)

        def objective(x):
            return objective_parts(x[:size], x[size:].reshape(samples, 2))

        cr, cc, colors = hessian_pattern(layout)
        self.color_count = int(colors.max()) + 1 if len(colors) else 0
        seeds = jnp.asarray(np.eye(self.color_count)[colors].T)
        mapping = np.full((ndim + 1) * count, -1, dtype=int)
        mapping[layout.control_indices] = np.arange(len(layout.control_indices))
        mapping[ndim * count + layout.weight_indices] = len(
            layout.control_indices
        ) + np.arange(len(layout.weight_indices))
        local_full_columns = np.concatenate(
            (
                (ndim * indices[:, :, None] + np.arange(ndim)).reshape(
                    samples, local_control_size
                ),
                ndim * count + indices,
            ),
            axis=1,
        )
        local_columns = mapping[local_full_columns]
        local_valid = np.concatenate(
            (np.repeat(valid, ndim, axis=1), valid), axis=1
        ) & (local_columns >= 0)
        mixed_mask = np.repeat(local_valid[:, :, None], 2, axis=2).ravel()
        mixed_rows = np.broadcast_to(
            size + 2 * np.arange(samples)[:, None, None] + np.arange(2),
            (samples, local_size, 2),
        ).ravel()[mixed_mask]
        mixed_cols = np.broadcast_to(
            local_columns[:, :, None], (samples, local_size, 2)
        ).ravel()[mixed_mask]
        pr = (size + 2 * np.arange(samples)[:, None] + [0, 1, 1]).ravel()
        pc = (size + 2 * np.arange(samples)[:, None] + [0, 0, 1]).ravel()
        self.rows, self.columns = np.r_[cr, mixed_rows, pr], np.r_[cc, mixed_cols, pc]
        cri, decode = jnp.asarray(cr), jnp.asarray(colors[cc])
        mask_indices = jnp.asarray(np.flatnonzero(mixed_mask))
        mixed = jax.jacfwd(jax.grad(point_loss, 0), 1)
        parameter_hessian = jax.hessian(point_loss, 1)

        def sparse_hessian(x):
            surface_variables, parameters = x[:size], x[size:].reshape(samples, 2)
            if self.color_count:
                _, product = jax.linearize(
                    lambda point: jax.grad(objective_parts, 0)(point, parameters),
                    surface_variables,
                )
                compressed = jax.lax.map(product, seeds, batch_size=4)
                block = compressed[decode, cri]
            else:
                block = jnp.empty((0,), dtype=x.dtype)
            local = local_values(surface_variables)
            coupling = jax.vmap(mixed)(
                local, parameters, self._target, jb, je, jvalid
            ).ravel()[mask_indices]
            diagonal = jax.vmap(parameter_hessian)(
                local, parameters, self._target, jb, je, jvalid
            )
            return jnp.concatenate(
                (
                    block,
                    coupling,
                    diagonal[:, jnp.array([0, 1, 1]), jnp.array([0, 0, 1])].ravel(),
                )
            )

        compiled = self._compile(
            [objective, jax.value_and_grad(objective), sparse_hessian],
            (jnp.asarray(self.x0),),
        )
        self._value, self._value_gradient, self._hessian = compiled
        self._finish_setup(begun)

    def parameters(self, x):
        return np.asarray(x[self.surface_size :]).reshape(-1, 2).copy()

    def set_parameters(self, parameters):
        raise ValueError("Reconstruct a joint problem to recenter its support windows")
