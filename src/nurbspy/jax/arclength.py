"""Knot-aware JAX arc length and reusable integration tables.

Adaptive quadrature is the general length method; an ODE alternative remains
available. Both use public solver APIs and expose unsuccessful termination.
Prepared Gauss tables are optional: their resolution must be checked on the
intended geometry family. Dense inverse maps use a reciprocal ODE on each knot
span, with no iterative quadrature in a query. See the usage guide for numerical
assumptions, differentiation, tolerances and table invalidation.
"""

import operator as _operator
from typing import NamedTuple

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import quadax as qdx

from .prepared import (
    PreparedCurveBasis as _PreparedCurveBasis,
    prepare_curve as _prepare_curve,
)

__all__ = ["ArcLengthInfo", "ArcLengthMap", "PreparedArcLength", "prepare_arclength"]


class ArcLengthInfo(NamedTuple):
    """Common value/work diagnostics; an unavailable estimate is NaN.

    Adaptive statuses 0..8 are Quadax's public severity enum, not bit flags.
    This wrapper adds 16=invalid geometry/options and 32=nonfinite sampled
    speed/value. ODE adds 64=unsuccessful solver termination. Fixed rules do not
    return this type: no tolerance or error estimator is implied by a rule.
    """

    value: jax.Array
    error_estimate: jax.Array
    evaluations: jax.Array
    status: jax.Array
    intervals: jax.Array


@jax.custom_jvp
def _speed_norm(derivative):
    """Euclidean speed; explicitly select the zero subgradient at zero speed.

    Length is not generally differentiable at a collapsed curve. Returning this
    finite convention does not establish existence of a unique derivative.
    """
    return jnp.sqrt(jnp.sum(derivative**2, axis=0))


@_speed_norm.defjvp
def __speed_norm_jvp(primals, tangents):
    (derivative,), (tangent,) = primals, tangents
    speed = _speed_norm(derivative)
    denominator = jnp.where(speed > 0, speed, 1.0)
    derivative_speed = jnp.where(
        speed > 0, jnp.sum(derivative * tangent, axis=0) / denominator, 0.0
    )
    return speed, derivative_speed


def _span_bounds(curve, u1=0.0, u2=1.0):
    """Static-shape clipped span bounds, oriented sign, and geometry validity.

    Repeated/clipped zero-width spans are retained to keep the JAX shape static.
    Public Quadax ignores them. Explicit limits preserve endpoint Leibniz
    derivatives; differentiating a clip at equality would choose half of one.
    """
    if any(jnp.iscomplexobj(x) for x in (curve.P, curve.W, curve.U, u1, u2)):
        raise ValueError("Arc length requires real geometry, knots and limits")
    dtype = jnp.result_type(curve.P, curve.W, curve.U, 1.0)
    u1, u2 = jnp.asarray(u1, dtype=dtype), jnp.asarray(u2, dtype=dtype)
    lo, hi = jnp.minimum(u1, u2), jnp.maximum(u1, u2)
    knots = jnp.clip(curve.U[curve.p : len(curve.U) - curve.p], lo, hi)
    knots = knots.at[0].set(lo).at[-1].set(hi)
    domain_lo, domain_hi = curve.U[curve.p], curve.U[-curve.p - 1]
    multiplicity = jnp.sum(curve.U[:, None] == curve.U[None, :], axis=1)
    interior = (curve.U > domain_lo) & (curve.U < domain_hi)
    valid = (
        (curve.p >= 0)
        & jnp.isfinite(u1) & jnp.isfinite(u2)
        & (domain_hi > domain_lo)
        & (lo >= domain_lo) & (hi <= domain_hi)
        & jnp.all(jnp.isfinite(curve.U)) & jnp.all(jnp.diff(curve.U) >= 0)
        & jnp.all(jnp.isfinite(curve.P)) & jnp.all(jnp.isfinite(curve.W))
        & jnp.all(curve.W > 0)
        & jnp.all(jnp.where(interior, multiplicity <= curve.p, True))
    )
    return knots, jnp.where(u2 >= u1, 1.0, -1.0), valid


def _normalised_geometry(curve, u1, u2, orient_endpoint):
    """Work in [0,1] so offsets in a translated knot domain do not hide layers.

    The entire curve is reflected when the first endpoint has the larger
    weight. This places the tested endpoint layer near zero, where tiny float64
    coordinates remain representable. It is a heuristic for positive rational
    endpoint layers, not a guarantee for arbitrary interior sharp features.
    """
    domain_lo, domain_hi = curve.U[curve.p], curve.U[-curve.p - 1]
    width = domain_hi - domain_lo
    safe_width = jnp.where(width > 0, width, 1.0)
    knots = (curve.U - domain_lo) / safe_width
    a, b = (u1 - domain_lo) / safe_width, (u2 - domain_lo) / safe_width
    reverse = (curve.W[0] > curve.W[-1]) if orient_endpoint else jnp.array(False)
    maximum_weight = jnp.max(jnp.abs(curve.W))
    weights = curve.W / jnp.where(maximum_weight > 0, maximum_weight, 1.0)
    points = curve.P.astype(jnp.result_type(curve.P, curve.W, 1.0))
    normalised = type(curve)(
        jnp.where(reverse, points[:, ::-1], points),
        jnp.where(reverse, weights[::-1], weights), degree=curve.p,
        knots=jnp.where(reverse, 1.0 - knots[::-1], knots),
    )
    return normalised, jnp.where(reverse, 1.0 - b, a), jnp.where(reverse, 1.0 - a, b)


def _graded_breakpoints(bounds, ratio):
    """Seed logarithmic fractions for the demonstrated rational endpoint layer.

    For weights [1,R], the line speed contains a width O(1/R) layer that a
    conventional unsampled error estimator can entirely miss. Nine logarithmic
    fractions at each span endpoint address that case. Extremely small panels
    can still exhaust floating-point resolution and must report failure.
    """
    left = 0.5 * ratio ** (-jnp.linspace(1.0, 0.0, 9))
    fractions = jnp.sort(jnp.concatenate((jnp.array([0.0, 1.0]), left, 1.0 - left)))
    panels = bounds[:-1, None] + jnp.diff(bounds)[:, None] * fractions
    return jnp.sort(panels.ravel())


def _adaptive_length(curve, u1=0.0, u2=1.0, *, atol=1e-6, rtol=1e-6,
                    max_intervals=512, grade_weights=True, orient_endpoint=True,
                    adjoint="direct"):
    """Knot-aware public GK21 with independent absolute/relative tolerances.

    Geometry/limits are normalized before evaluation. Large positive weight
    ratios (>1e4) activate an endpoint grading heuristic. Set grade_weights=False
    to demonstrate the unsampled-layer limitation of ordinary adaptivity.

    DirectAdjoint differentiates the accepted discretization; LeibnizAdjoint
    gives the derivative its own adaptive solve. Neither an integral's error
    estimate nor status 0 is a gradient-error certificate. The nonfinite-speed
    check probes mesh ends/midpoints; it cannot inspect every internal sample
    because Quadax intentionally masks nonfinite integrand samples.
    """
    max_intervals = _operator.index(max_intervals)
    if max_intervals < 1:
        raise ValueError("max_intervals must be a positive integer")
    _, _, original_valid = _span_bounds(curve, u1, u2)
    curve, a, b = _normalised_geometry(curve, jnp.asarray(u1), jnp.asarray(u2), orient_endpoint)
    bounds, sign, valid = _span_bounds(curve, a, b)
    valid = (valid & original_valid & jnp.isfinite(atol) & jnp.isfinite(rtol)
             & (atol >= 0) & (rtol >= 0) & ((atol > 0) | (rtol > 0)))
    if adjoint == "direct":
        differentiation = qdx.DirectAdjoint()
    elif adjoint == "leibniz":
        differentiation = qdx.LeibnizAdjoint()
    else:
        raise ValueError("adjoint must be 'direct' or 'leibniz'")

    def speed(x):
        return jnp.squeeze(_speed_norm(curve.get_derivative(x, order=1)))

    def integrate(points):
        if len(points) - 1 > max_intervals:
            # A static conservative bound: repeated points cost no work in
            # Quadax, but require a larger workspace for its seed array.
            return ArcLengthInfo(jnp.array(jnp.nan), jnp.array(jnp.inf),
                                jnp.array(0), jnp.array(1), jnp.array(len(points) - 1))
        value, info = qdx.quadgk(
            speed, points, epsabs=atol, epsrel=rtol, max_ninter=max_intervals,
            full_output=True, adjoint=differentiation, extrapolate=False, throw=False,
        )
        probes = jnp.concatenate((points, (points[:-1] + points[1:]) / 2))
        finite = jnp.all(jnp.isfinite(speed(probes))) & jnp.isfinite(value)
        status = jnp.where(valid, jnp.where(finite, info.status, 32), 16)
        return ArcLengthInfo(sign * value, jax.lax.stop_gradient(info.err),
                            info.neval, status, info.info["ninter"])

    def nonzero(_):
        if not grade_weights:
            return integrate(bounds)
        ratio = jnp.max(curve.W) / jnp.min(curve.W)
        return jax.lax.cond(ratio > 1e4,
                            lambda _: integrate(_graded_breakpoints(bounds, ratio)),
                            lambda _: integrate(bounds), operand=None)

    def zero(_):
        speed_value = speed(a)
        value = (b - a) * speed_value
        return ArcLengthInfo(value, jnp.array(0.0), jnp.array(1),
                            jnp.where(jnp.isfinite(speed_value), 0, 32), jnp.array(0))

    def invalid(_):
        return ArcLengthInfo(jnp.array(jnp.nan), jnp.array(jnp.inf),
                            jnp.array(0), jnp.array(16), jnp.array(0))

    return jax.lax.cond(valid,
        lambda _: jax.lax.cond(a == b, zero, nonzero, operand=None), invalid, operand=None)


def _ode_length(curve, u1=0.0, u2=1.0, *, atol=1e-6, rtol=1e-6,
               max_steps=4096):
    """Forward speed ODE with public forced steps at all interior knots.

    Smooth knots force a step boundary. Knots of multiplicity >= degree are
    marked as jumps, so Diffrax recomputes the vector field instead of reusing
    an FSAL stage across a possible C0 speed jump. Treating every smooth knot as
    a jump was measurably worse on the narrow-span case in this study.
    Attempted step counts are exposed; Diffrax does not expose exact field
    evaluation counts here, so evaluations=-1. No global error is invented.
    Reversed limits give a signed integral, and zero intervals retain the
    endpoint derivatives of that integral.
    """
    max_steps = _operator.index(max_steps)
    if max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    _, _, original_valid = _span_bounds(curve, u1, u2)
    curve, a, b = _normalised_geometry(curve, jnp.asarray(u1), jnp.asarray(u2), False)
    points, sign, valid = _span_bounds(curve, a, b)
    lo, hi = jnp.minimum(a, b), jnp.maximum(a, b)
    valid = (valid & original_valid & jnp.isfinite(atol) & jnp.isfinite(rtol)
             & (atol >= 0) & (rtol >= 0) & ((atol > 0) | (rtol > 0)))
    controller = dfx.PIDController(atol=atol, rtol=rtol)
    if len(points) > 2:
        interior = points[1:-1]
        multiplicity = jnp.sum(interior[:, None] == curve.U[None, :], axis=1)
        active = (interior > lo) & (interior < hi)
        # Inactive entries become +inf, retaining static JAX shapes (including
        # partial limits) without empty-controller arrays or host np.unique.
        steps = jnp.sort(jnp.where(active & (multiplicity < curve.p), interior, jnp.inf))
        jumps = jnp.sort(jnp.where(active & (multiplicity >= curve.p), interior, jnp.inf))
        controller = dfx.ClipStepSizeController(controller, step_ts=steps, jump_ts=jumps)

    def speed(x, y, args):
        return _speed_norm(curve.get_derivative(x, order=1))[0]

    def nonzero(_):
        solution = dfx.diffeqsolve(
            dfx.ODETerm(speed), dfx.Dopri5(), t0=lo, t1=hi, y0=jnp.array(0.0),
            dt0=jnp.minimum(1e-6, hi - lo), stepsize_controller=controller,
            adjoint=dfx.DirectAdjoint(), saveat=dfx.SaveAt(t1=True),
            max_steps=max_steps, throw=False,
        )
        value = solution.ys[0]
        finite = jnp.isfinite(value)
        # Seven stages for the initial step; FSAL saves one per subsequent
        # ordinary step, but forced jumps invalidate it. Do not invent neval.
        status = jnp.where(valid, jnp.where(finite,
            jnp.where(solution.result == dfx.RESULTS.successful, 0, 64), 32), 16)
        return ArcLengthInfo(sign * value, jnp.array(jnp.nan), jnp.array(-1),
                            status, solution.stats["num_steps"])

    def zero(_):
        speed_value = speed(a, None, None)
        value = (b - a) * speed_value
        return ArcLengthInfo(value, jnp.array(jnp.nan), jnp.array(1),
                            jnp.where(jnp.isfinite(speed_value), 0, 32), jnp.array(0))

    def invalid(_):
        return ArcLengthInfo(jnp.array(jnp.nan), jnp.array(jnp.nan),
                            jnp.array(0), jnp.array(16), jnp.array(0))

    return jax.lax.cond(valid,
        lambda _: jax.lax.cond(a == b, zero, nonzero, operand=None), invalid, operand=None)


def _spanwise_nodes(curve, u1=0.0, u2=1.0, *, order=16, subdivisions=1):
    """Gauss nodes and oriented weights on every clipped knot span."""
    order, subdivisions = _operator.index(order), _operator.index(subdivisions)
    if order < 2 or subdivisions < 1:
        raise ValueError("order >= 2 and subdivisions >= 1 are required")
    bounds, sign, _ = _span_bounds(curve, u1, u2)
    fractions = jnp.linspace(0.0, 1.0, subdivisions + 1)
    panels = bounds[:-1, None] + jnp.diff(bounds)[:, None] * fractions
    left, right = panels[:, :-1].ravel(), panels[:, 1:].ravel()
    x, w = np.polynomial.legendre.leggauss(order)
    nodes = (left[:, None] + right[:, None]) / 2 + (right - left)[:, None] * jnp.asarray(x) / 2
    weights = sign * (right - left)[:, None] * jnp.asarray(w) / 2
    return nodes.ravel(), weights.ravel()


class PreparedArcLength(eqx.Module):
    """Fixed Gauss nodes and basis tables reusable with changing P and W.

    Rebuild for changed knots, degree, limits, order or subdivisions. The object
    stores polynomial basis tables, not a snapshot of rational geometry. Fixed
    rules have no convergence guarantee; compare with adaptive integration on
    the intended design family before using them in an objective.
    """

    basis: _PreparedCurveBasis
    weights: jax.Array

    @eqx.filter_jit
    def value(self, P, W=None):
        """Return a differentiable length; omitted weights mean unit weights."""
        P, W = jnp.asarray(P), None if W is None else jnp.asarray(W)
        if jnp.iscomplexobj(P) or (W is not None and jnp.iscomplexobj(W)):
            raise ValueError("Prepared arc length requires real geometry and weights")
        value = jnp.sum(self.weights * _speed_norm(self.basis.get_derivative(P, W, order=1)))
        valid = jnp.all(jnp.isfinite(P)) & jnp.isfinite(value)
        if W is not None:
            valid = valid & jnp.all(jnp.isfinite(W)) & jnp.all(W > 0)
        return eqx.error_if(value, ~valid, "Prepared arc length requires finite geometry and positive weights")


@eqx.filter_jit
def prepare_arclength(curve, u1=0.0, u2=1.0, *, order=16, subdivisions=1):
    """Prepare basis tables; rebuild for new knots/limits/order.

    Build inside an AD transform for knot/limit sensitivities. At a zero-width
    interval fixed quadrature currently does not promise endpoint AD semantics.
    Even GL16/GL32 agreement can miss an extreme rational endpoint layer, so
    validate the expected geometry family with an independent adaptive/reference
    calculation rather than an order comparison alone.
    """
    _, _, valid = _span_bounds(curve, u1, u2)
    nodes, weights = _spanwise_nodes(curve, u1, u2, order=order, subdivisions=subdivisions)
    # Tables depend on P's shape, not its values. Tie the guard to a returned
    # numerical array so JIT cannot discard it as an unused geometry check.
    weights = eqx.error_if(weights, ~valid,
        "Arc length requires a continuous real curve, positive weights and in-domain limits")
    return PreparedArcLength(_prepare_curve(curve, nodes, max_order=1), weights)


def _checked_value(info):
    return eqx.error_if(info.value, info.status != 0,
                        "Arc-length integration failed or inputs are invalid. "
                        "Use return_info=True to inspect status/error, or increase the integration budget.")


class ArcLengthMap(eqx.Module):
    """Dense reciprocal-ODE data behind a curve's compiled inverse callable.

    Queries accept scalars or arrays, preserve shape, and support JAX AD on
    regular curves. Changing geometry requires rebuilding the map; keep its
    construction inside an AD transform for geometry sensitivities. Each knot
    span starts at its exact parameter and its dense endpoint is normalized to
    the other exact endpoint. This prevents drift across speed jumps.
    """

    solutions: dfx.Solution
    bounds: jax.Array
    cumulative: jax.Array
    end_values: jax.Array

    @property
    def total_length(self):
        return self.cumulative[-1]

    @eqx.filter_jit
    def __call__(self, s):
        s = jnp.asarray(s, dtype=self.bounds.dtype)
        s = eqx.error_if(s, jnp.any(~jnp.isfinite(s) | (s < 0) | (s > self.total_length)),
                         "Arc-length map queries must lie between zero and the total length")

        def evaluate_one(distance):
            panel = jnp.clip(jnp.searchsorted(self.cumulative, distance, side="right")-1,
                             0, self.bounds.size-2)
            length = self.cumulative[panel+1]-self.cumulative[panel]
            time = (distance-self.cumulative[panel])/length
            solution = jax.tree.map(lambda value: value[panel], self.solutions)
            x = solution.evaluate(time)/self.end_values[panel]
            return self.bounds[panel] + jnp.diff(self.bounds)[panel]*x

        values = jax.vmap(evaluate_one)(s.reshape(-1)).reshape(s.shape)
        # Exact endpoint values, preserving the interpolation's one-sided JVP.
        correction = jnp.where(s == 0, self.bounds[0]-values,
                               jnp.where(s == self.total_length, self.bounds[-1]-values, 0.0))
        return values + jax.lax.stop_gradient(correction)

    evaluate = __call__


@eqx.filter_jit
def _prepare_map(curve, *, tol=1e-6, max_steps=512, steps_per_span=16,
                 max_intervals=512):
    """Compile reusable span solves; no host loops, roots or hidden speed floor.

    Both ODE time and its dependent parameter are normalized within a span.
    Repeated zero-width spans are retained for static JAX shapes; their dummy
    solutions are never selected by a valid query. Dense storage is bounded by
    max_steps per span. Local tolerances do not certify global inverse accuracy.
    """
    max_steps, steps_per_span = _operator.index(max_steps), _operator.index(steps_per_span)
    if max_steps < 1 or steps_per_span < 1:
        raise ValueError("max_steps and steps_per_span must be positive integers")
    bounds, _, valid = _span_bounds(curve, curve.U[curve.p], curve.U[-curve.p-1])
    valid = valid & jnp.isfinite(tol) & (tol > 0)
    curve = eqx.tree_at(lambda c: c.P, curve, eqx.error_if(
        curve.P, ~valid, "Arc-length maps require finite real geometry and positive weights"))
    a, b = bounds[:-1], bounds[1:]
    widths = b-a
    active = widths > 0

    # Explicit left limits at each right endpoint avoid sampling the following
    # span's derivative at a C0 knot. Stop only the unrepresentable ulp shift,
    # retaining derivatives with respect to the actual knot positions.
    frozen_a, frozen_b = jax.lax.stop_gradient(a), jax.lax.stop_gradient(b)
    right = b-(frozen_b-jnp.nextafter(frozen_b, frozen_a))
    samples = a[:, None]+(right-a)[:, None]*jnp.linspace(0.0, 1.0, 17)
    speeds = _speed_norm(curve.get_derivative(samples.ravel(), 1)).reshape(samples.shape)
    regular = jnp.all(jnp.where(active[:, None], jnp.isfinite(speeds) & (speeds > 0), True))
    curve = eqx.tree_at(lambda c: c.P, curve, eqx.error_if(
        curve.P, ~regular, "The reciprocal ODE requires positive speed; stationary/flat curves need a different inverse method"))

    infos = jax.vmap(lambda lo, hi: _adaptive_length(
        curve, lo, hi, atol=tol/max(1, len(a)), rtol=tol,
        max_intervals=max_intervals))(a, b)
    lengths = eqx.error_if(infos.value,
        jnp.any(infos.status != 0) | jnp.any(active & (infos.value <= 0)),
        "Arc-length map construction requires converged positive knot-span lengths")
    cumulative = jnp.concatenate((jnp.zeros(1, dtype=lengths.dtype), jnp.cumsum(lengths)))
    cumulative = eqx.error_if(cumulative, cumulative[-1] <= 0,
                              "A zero-length curve has no unique arc-length inverse")

    def solve(lo, hi, right_limit, length):
        width = hi-lo
        def field(t, x, args):
            u = jnp.minimum(lo+width*jnp.clip(x, 0, 1), right_limit)
            speed = _speed_norm(curve.get_derivative(u, 1))[0]
            # Only the artificial zero-width spans use the dummy x'=1.
            denominator = jnp.where(width > 0, width*speed, 1.0)
            return jnp.where(width > 0, length/denominator, 1.0)
        return dfx.diffeqsolve(
            dfx.ODETerm(field), dfx.Dopri5(), t0=0.0, t1=1.0,
            y0=jnp.array(0.0), dt0=1e-4,
            stepsize_controller=dfx.PIDController(atol=tol, rtol=tol,
                                                 dtmax=1.0/steps_per_span),
            saveat=dfx.SaveAt(t1=True, dense=True), adjoint=dfx.DirectAdjoint(),
            max_steps=max_steps, throw=False,
        )

    solutions = jax.vmap(solve)(a, b, right, lengths)
    end_values = solutions.ys[:, 0]
    probes = jax.vmap(lambda solution: jax.vmap(solution.evaluate)(jnp.linspace(0, 1, 33)))(solutions)
    good = (jnp.all(solutions.result == dfx.RESULTS.successful)
            & jnp.all(jnp.isfinite(probes)) & jnp.all(jnp.diff(probes, axis=1) > 0)
            & jnp.all(end_values > 0))
    end_values = eqx.error_if(end_values, ~good,
                              "Arc-length inverse ODE failed or its sampled dense output is not monotone")
    return ArcLengthMap(solutions, bounds, cumulative, end_values)
