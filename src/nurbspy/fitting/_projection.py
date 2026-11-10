"""Batched adapter for the existing public JAX surface projector."""

from time import perf_counter
import numpy as np
from ._common import parameter_domain


class SurfaceProjector:
    """Reuse checked library projection, rebuilding samples when geometry changes."""

    def __init__(
        self,
        initial,
        points,
        *,
        initial_parameters=None,
        tolerance=1e-9,
        maxiter=200,
        batch_size=64,
        samples_per_span=8
    ):
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        from ..jax import prepare_surface_projection, PreparedSurfaceProjection

        self.points = np.array(points, dtype=float, copy=True)
        self.initial, self.batch_size = initial, min(batch_size, len(points))
        self.domain = parameter_domain(initial)
        if initial_parameters is None:
            # Seed candidates are spread over the actual domain; query x/y are
            # never interpreted as parameter coordinates.
            count = len(points)
            fraction = np.column_stack(
                (
                    (np.arange(count) + 0.5) / count,
                    (np.arange(count) * 0.6180339887498949) % 1,
                )
            )
            initial_parameters = self.domain[0] + fraction * (
                self.domain[1] - self.domain[0]
            )
        self.default_parameters = np.array(initial_parameters, copy=True)
        padding = (-len(points)) % self.batch_size
        padded = np.pad(self.points, ((0, padding), (0, 0)), mode="edge")
        self.batches = [
            jnp.asarray(batch.T)
            for batch in padded.reshape(-1, self.batch_size, points.shape[1])
        ]

        @eqx.filter_jit
        def prepare(surface, parameters):
            prepared = prepare_surface_projection(
                surface, samples_per_span=samples_per_span
            )
            extra = surface.get_value(parameters[:, 0], parameters[:, 1])
            return PreparedSurfaceProjection(
                surface=surface,
                sample_parameters=jnp.concatenate(
                    (prepared.sample_parameters, parameters)
                ),
                sampled_points=jnp.concatenate(
                    (prepared.sampled_points, extra), axis=1
                ),
                samples_per_span=samples_per_span,
            )

        @eqx.filter_jit
        def project(prepared, queries):
            return prepared.project_points(
                queries, maxiter=maxiter, ftol=1e-12, gtol=tolerance, xtol=1e-12
            )

        surface = self._surface(initial)
        parameters = jnp.asarray(initial_parameters)
        begun = perf_counter()
        self._prepare = prepare.lower(surface, parameters).compile()
        self.compile_seconds = perf_counter() - begun
        begun = perf_counter()
        prepared = jax.block_until_ready(self._prepare(surface, parameters))
        self.warmup_seconds = perf_counter() - begun
        begun = perf_counter()
        self._project = project.lower(prepared, self.batches[0]).compile()
        self.compile_seconds += perf_counter() - begun
        begun = perf_counter()
        known = prepared.sampled_points[:, : self.batch_size]
        if known.shape[1] < self.batch_size:
            known = jnp.pad(
                known, ((0, 0), (0, self.batch_size - known.shape[1])), mode="edge"
            )
        jax.block_until_ready(self._project(prepared, known))
        self.warmup_seconds += perf_counter() - begun

    @staticmethod
    def _surface(model):
        from ..jax import NurbsSurface

        return NurbsSurface(
            np.asarray(model.P),
            weights=np.asarray(model.W),
            u_degree=int(model.p),
            v_degree=int(model.q),
            u_knots=np.asarray(model.U),
            v_knots=np.asarray(model.V),
        )

    def project(self, model, parameters=None):
        import jax
        import jax.numpy as jnp
        from ._common import evaluate

        if parameters is None:
            parameters = self.default_parameters
        parameters = np.clip(parameters, *self.domain)
        begun = perf_counter()
        prepared = jax.block_until_ready(
            self._prepare(self._surface(model), jnp.asarray(parameters))
        )
        uv = np.concatenate(
            [np.asarray(self._project(prepared, batch)).T for batch in self.batches]
        )[: len(self.points)]
        elapsed = perf_counter() - begun
        prediction = evaluate(
            model,
            np.asarray(model.P).reshape(model.P.shape[0], -1).T,
            np.asarray(model.W).ravel(),
            uv,
        )
        distances = np.linalg.norm(prediction - self.points, axis=1)
        return (
            uv,
            distances,
            dict(
                success=True,
                solve_seconds=elapsed,
                backend="NurbsPy JAX prepared surface projection",
                queries=len(self.points),
            ),
        )
