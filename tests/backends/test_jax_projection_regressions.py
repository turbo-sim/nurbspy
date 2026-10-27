"""Sampled single-start projection: difficult geometry, domains, and derivatives.

References use SciPy's rational B-splines or closed-form geometry, rather than
another invocation of the projection implementation under test.
"""

from importlib.util import find_spec

import numpy as np
from numpy.polynomial import Polynomial
from numpy.testing import assert_allclose
import pytest
from scipy.interpolate import BSpline
from scipy.optimize import brentq, root

if find_spec("jax") is None:
    pytest.skip("Install nurbspy[jax] for projection checks.", allow_module_level=True)

import jax
import jax.numpy as jnp
import nurbspy.jax as nrb


def _rational_curve(points, weights, knots, degree):
    """Evaluate position and its first two derivatives independently of JAX."""
    basis = BSpline(knots, np.eye(points.shape[1]), degree)
    homogeneous = np.vstack((points * weights, weights))

    def evaluate(parameter):
        rows = [
            homogeneous @ basis(parameter, nu=order)
            if order <= degree
            else np.zeros(points.shape[0] + 1)
            for order in range(3)
        ]
        value = rows[0][:-1] / rows[0][-1]
        first = (rows[1][:-1] - value * rows[1][-1]) / rows[0][-1]
        second = (
            rows[2][:-1] - 2 * first * rows[1][-1] - value * rows[2][-1]
        ) / rows[0][-1]
        return value, first, second

    return evaluate


def _reference_distances(points, weights, knots, degree, queries):
    """Compare all bracketed stationary minima and knot-boundary candidates.

    These selected smooth test curves have no unresolved roots on this much
    denser reference grid. This is a regression reference, not a general global
    optimality certificate for arbitrary NURBS geometry.
    """
    evaluate = _rational_curve(points, weights, knots, degree)
    boundaries = np.unique(knots[degree : -degree])
    fractions = np.unique(np.r_[np.linspace(0.0, 1.0, 1025), 10.0 ** np.linspace(-10, -1, 32)])
    fractions = np.unique(np.r_[fractions, 1 - fractions])
    grid = np.unique(
        np.concatenate(
            [a + (b - a) * fractions for a, b in zip(boundaries[:-1], boundaries[1:])]
        )
    )
    values = np.array([evaluate(parameter)[0] for parameter in grid])
    tangents = np.array([evaluate(parameter)[1] for parameter in grid])
    distances = []
    for query in queries.T:
        residuals = values - query
        gradients = np.einsum("ij,ij->i", residuals, tangents)
        candidates = list(boundaries)
        candidates.extend(grid[gradients == 0])
        for i in np.flatnonzero((gradients[:-1] < 0) & (gradients[1:] > 0)):
            candidates.append(
                brentq(
                    lambda u: np.dot(evaluate(u)[0] - query, evaluate(u)[1]),
                    grid[i],
                    grid[i + 1],
                    xtol=5e-15,
                    rtol=1e-14,
                )
            )
        distances.append(
            min(np.sum((evaluate(u)[0] - query) ** 2) for u in candidates)
        )
    return np.asarray(distances)


@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reversed"])
def test_strongly_weighted_line_resolves_endpoint_layer(reverse):
    """Huge negative Hessians and tiny feasible steps still reach the line minimum."""
    points = np.array([[0.0, 1.0], [0.0, 0.0]])
    weights = np.array([1.0, 1e8])
    if reverse:
        points, weights = points[:, ::-1].copy(), weights[::-1].copy()
    curve = nrb.NurbsCurve(points, weights)
    queries = jnp.array([[0.25, 0.5, 0.75, 0.99], [0.1, -0.2, 0.1, 0.2]])
    # Mirroring compresses the layer against u=1, whose float64 spacing imposes
    # a few 1e-9 units of coordinate error for this unit-size test geometry.
    actual = curve.project_points(queries, tol=1e-8 if reverse else 1e-9)
    expected = np.asarray(queries[0]) / (
        1e8 * (1 - np.asarray(queries[0])) + np.asarray(queries[0])
    )
    if reverse:
        expected = 1 - expected
    assert_allclose(actual, expected, rtol=1e-5, atol=2e-14)
    # Position is a more meaningful accuracy measure in this compressed domain.
    assert_allclose(curve.get_value(actual)[0], queries[0], rtol=0, atol=3e-8)
    distances = jnp.sum((curve.get_value(actual) - queries) ** 2, axis=0)
    assert_allclose(distances, queries[1] ** 2, rtol=0, atol=1e-12)


@pytest.mark.parametrize("weighted", [False, True], ids=["folded", "rational"])
def test_sampled_initialization_selects_competing_curve_basins(weighted):
    """Actual curve samples avoid misleading control-point-index initialization."""
    points = np.array(
        [[0.0, 0.6, 0.2, 0.9, 0.3, 1.1, 0.5, 1.0],
         [0.0, 1.4, -1.2, 1.1, -1.3, 1.2, -0.9, 0.2]]
    )
    rng = np.random.default_rng(27)
    rng.uniform(-0.4, 0.8, 16)  # Preserve the deterministic study's query sequence.
    folded_queries = np.vstack((rng.uniform(-0.2, 1.2, 16), rng.uniform(-0.7, 0.9, 16)))
    rational_queries = np.vstack((rng.uniform(-0.2, 1.2, 16), rng.uniform(-1.0, 1.0, 16)))
    if weighted:
        points = points[:, :6]
        weights = np.array([1.0, 0.02, 80.0, 0.04, 25.0, 1.0])
        knots = np.r_[np.zeros(4), [0.12, 0.85], np.ones(4)]
        queries = rational_queries[:, [0, 3, 7, 11, 15]]
    else:
        weights = np.ones(8)
        knots = np.r_[np.zeros(4), [0.04, 0.18, 0.66, 0.93], np.ones(4)]
        queries = folded_queries[:, [0, 3, 7, 11, 15]]
    curve = nrb.NurbsCurve(points, weights, 3, knots)
    parameters = curve.project_points(queries, tol=1e-9)
    actual = jnp.sum((curve.get_value(parameters) - queries) ** 2, axis=0)
    expected = _reference_distances(points, weights, knots, 3, queries)
    assert_allclose(actual, expected, rtol=1e-8, atol=2e-10)


def test_curve_projection_respects_nonclamped_active_domain():
    """The search interval is [U[p], U[-p-1]], rather than [0, 1]."""
    points = np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, -0.5, 1.0, 0.0]])
    knots = np.arange(8, dtype=float)
    queries = np.array([[0.2, 1.5, 3.8, 2.5], [0.3, -0.1, 0.4, 0.2]])
    curve = nrb.NurbsCurve(points, np.ones(5), 2, knots)
    parameters = curve.project_points(queries, tol=1e-9)
    assert np.all((np.asarray(parameters) >= 2) & (np.asarray(parameters) <= 5))
    actual = jnp.sum((curve.get_value(parameters) - queries) ** 2, axis=0)
    expected = _reference_distances(points, np.ones(5), knots, 2, queries)
    assert_allclose(actual, expected, atol=2e-10)


def test_piecewise_linear_curve_keeps_exact_corner_minimum():
    """A nonsmooth knot minimum satisfies one-sided, not smooth, stationarity."""
    curve = nrb.NurbsCurve(
        jnp.array([[0.0, 1.0, 2.0], [0.0, 1.0, 0.0]]),
        degree=1,
        knots=jnp.array([0.0, 0.0, 0.5, 1.0, 1.0]),
    )
    queries = jnp.array([[1.0, 0.8, 1.2], [1.5, 1.4, 1.4]])
    parameters = curve.project_points(queries, tol=1e-10)
    assert_allclose(parameters, 0.5, atol=1e-12)
    assert_allclose(curve.get_value(parameters), np.array([[1.0] * 3, [1.0] * 3]))


def test_collapsed_curve_keeps_finite_forward_projection():
    """Nonunique parameters still represent a valid closest point."""
    curve = nrb.NurbsCurve(jnp.zeros((2, 3)))
    queries = jnp.array([[1.0, 0.0], [0.2, 0.0]])
    parameters = curve.project_points(queries)
    assert np.all(np.isfinite(np.asarray(parameters)))
    assert np.all((np.asarray(parameters) >= 0) & (np.asarray(parameters) <= 1))
    assert_allclose(curve.get_value(parameters), np.zeros((2, 2)))


@pytest.mark.parametrize(
    "options",
    [
        {"tol": -1.0},
        {"gtol": -1.0},
        {"xtol": -1.0},
        {"ftol": -1.0},
        {"maxiter": -1},
        {"samples_per_span": 0},
        {"samples_per_span": 1.5},
    ],
)
def test_curve_projection_rejects_invalid_solver_options(options):
    """Invalid tolerances and static work budgets fail before solving."""
    curve = nrb.NurbsCurve(jnp.array([[0.0, 1.0], [0.0, 0.0]]))
    with pytest.raises((TypeError, ValueError)):
        curve.project_points(jnp.array([[0.3], [0.5]]), **options)


def test_curve_projection_rejects_wrong_query_shape_and_empty_geometry():
    """A point batch must have one coordinate row per spatial dimension."""
    curve = nrb.NurbsCurve(jnp.array([[0.0, 1.0], [0.0, 0.0]]))
    with pytest.raises(ValueError, match="shape"):
        curve.project_points(jnp.zeros((3, 2)))
    with pytest.raises(ValueError, match="nonempty"):
        nrb.NurbsCurve().project_points(jnp.zeros((2, 1)))


@pytest.mark.parametrize("weight", [0.0, -1.0, np.inf])
def test_curve_projection_rejects_invalid_rational_weights(weight):
    """Projection safeguards require finite, strictly positive denominators."""
    curve = nrb.NurbsCurve(
        jnp.array([[0.0, 1.0], [0.0, 0.0]]), jnp.array([1.0, weight])
    )
    with pytest.raises(Exception, match="weights"):
        np.asarray(curve.project_points(jnp.array([[0.3], [0.5]])))


def test_curve_tolerance_is_used_and_exhausted_iterations_are_reported():
    """A loose tolerance may accept a sample; a failed tight solve is explicit."""
    curve = nrb.NurbsCurve(jnp.array([[0.0, 1.0], [0.0, 0.0]]))
    query = jnp.array([[0.3], [0.5]])
    loose = curve.project_points(query, tol=1.0, samples_per_span=1, maxiter=0)
    assert np.all(np.isfinite(np.asarray(loose)))
    assert abs(float(loose[0]) - 0.3) > 0.01
    with pytest.raises(Exception, match="did not converge"):
        np.asarray(curve.project_points(query, tol=1e-12, samples_per_span=1, maxiter=0))
    # An explicit gtol takes precedence over the legacy tol argument.
    override = curve.project_points(
        query, tol=1e-12, gtol=1.0, samples_per_span=1, maxiter=0
    )
    assert_allclose(override, loose, rtol=0, atol=0)
    assert_allclose(curve.project_points(query, tol=1e-12, samples_per_span=1), [0.3], atol=1e-12)


@pytest.mark.parametrize("case", ["weak_bound", "corner", "c1_knot", "collapsed"])
@pytest.mark.parametrize("mode", [jax.jacfwd, jax.jacrev], ids=["forward", "reverse"])
def test_irregular_curve_minima_do_not_report_smooth_parameter_derivatives(case, mode):
    """Finite forward minima can have an undefined or unsupported smooth JVP."""
    if case == "weak_bound":
        curve = nrb.NurbsCurve(jnp.array([[0.0, 1.0], [0.0, 0.0]]))
        query = jnp.array([0.0, 0.5])
    elif case == "corner":
        curve = nrb.NurbsCurve(
            jnp.array([[0.0, 1.0, 2.0], [0.0, 1.0, 0.0]]),
            degree=1,
            knots=jnp.array([0.0, 0.0, 0.5, 1.0, 1.0]),
        )
        query = jnp.array([1.0, 1.5])
    elif case == "c1_knot":
        # The basis is only C1 at this knot. The documented derivative policy
        # is conservative even when a particular control net cancels the kink.
        curve = nrb.NurbsCurve(
            jnp.array([[0.0, 0.25, 0.75, 1.0], [0.0, 0.0, 0.0, 0.0]]),
            degree=2,
            knots=jnp.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]),
        )
        query = jnp.array([0.5, 0.2])
    else:
        curve = nrb.NurbsCurve(jnp.zeros((2, 3)))
        query = jnp.array([1.0, 0.2])

    def project(point):
        return curve.project_points(point[:, None])[0]

    assert np.isfinite(float(project(query)))
    assert np.all(np.isnan(np.asarray(jax.jit(mode(project))(query))))


@pytest.mark.parametrize("mode", [jax.jacfwd, jax.jacrev], ids=["forward", "reverse"])
def test_curve_projection_keeps_smooth_derivative_at_c2_internal_knot(mode):
    """A simple cubic knot is C2 and supports the exact implicit Hessian."""
    knots = jnp.array([0.0] * 4 + [0.5] + [1.0] * 4)
    greville = jnp.array([0.0, 1 / 6, 0.5, 5 / 6, 1.0])
    curve = nrb.NurbsCurve(jnp.stack((greville, jnp.zeros(5))), degree=3, knots=knots)

    def project(point):
        return curve.project_points(point[:, None])[0]

    query = jnp.array([0.5, 0.2])
    assert_allclose(project(query), 0.5, atol=1e-12)
    assert_allclose(jax.jit(mode(project))(query), [1.0, 0.0], atol=1e-10)


@pytest.mark.parametrize("mode", [jax.jacfwd, jax.jacrev], ids=["forward", "reverse"])
def test_curve_query_and_moving_domain_derivatives(mode):
    """Strict endpoints follow moving bounds; interior parameters follow the query."""
    points = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    queries = jnp.array([[0.3, -0.2, 1.2], [0.5, 0.5, 0.5]])
    knots = jnp.array([2.0, 2.0, 5.0, 5.0])

    def project(query):
        return nrb.NurbsCurve(points, degree=1, knots=knots).project_points(query)

    assert_allclose(project(queries), [2.9, 2.0, 5.0], atol=1e-10)
    expected = np.zeros((3, 2, 3))
    expected[0, 0, 0] = 3.0
    assert_allclose(jax.jit(mode(project))(queries), expected, atol=1e-10)

    def translate_domain(offset):
        return nrb.NurbsCurve(points, degree=1, knots=knots + offset).project_points(queries)

    assert_allclose(jax.jit(mode(translate_domain))(0.0), np.ones(3), atol=1e-10)


@pytest.mark.parametrize("mode", [jax.jacfwd, jax.jacrev], ids=["forward", "reverse"])
def test_curved_curve_query_derivative_includes_residual_curvature(mode):
    """The derivative of an off-curve minimizer uses more than the tangent Gramian."""
    curve = nrb.NurbsCurve(jnp.array([[0.0, 0.5, 1.0], [0.0, 0.0, 1.0]]))
    parameter = 0.31
    tangent = np.array([1.0, 2 * parameter])
    normal = np.array([-2 * parameter, 1.0]) / np.linalg.norm(tangent)
    query = np.array([parameter, parameter**2]) + 0.1 * normal
    hessian = np.dot(tangent, tangent) + 2 * (parameter**2 - query[1])

    def project(point):
        return curve.project_points(point[:, None], tol=1e-10)[0]

    assert_allclose(project(query), parameter, atol=1e-9)
    assert_allclose(jax.jit(mode(project))(query), tangent / hessian, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("argument", ["points", "weights"])
def test_curve_geometry_derivatives_use_exact_stationarity(argument):
    """Differentiate parameters themselves, including residual-curvature effects."""
    points = np.array([[0.0, 0.5, 1.0], [0.0, 2.0, 0.0]])
    weights = np.ones(3)
    knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    query = np.array([0.35, 1.3])
    variable = points if argument == "points" else weights
    direction = np.linspace(-0.2, 0.3, variable.size).reshape(variable.shape)

    def project(value):
        curve = nrb.NurbsCurve(
            value if argument == "points" else points,
            value if argument == "weights" else weights,
        )
        return curve.project_points(query[:, None], tol=1e-10)[0]

    parameter = float(project(variable))

    def reference(offset):
        evaluate = _rational_curve(
            points + offset * direction if argument == "points" else points,
            weights + offset * direction if argument == "weights" else weights,
            knots,
            2,
        )
        solution = root(
            lambda u: np.dot(evaluate(u[0])[0] - query, evaluate(u[0])[1]),
            [parameter],
            tol=1e-12,
        )
        assert_allclose(solution.fun, 0.0, atol=1e-11)
        return solution.x[0]

    step = 1e-5
    expected = (reference(step) - reference(-step)) / (2 * step)
    _, forward = jax.jit(lambda x, dx: jax.jvp(project, (x,), (dx,)))(
        jnp.asarray(variable), jnp.asarray(direction)
    )
    reverse = jnp.sum(jax.jit(jax.grad(project))(jnp.asarray(variable)) * direction)
    assert_allclose([forward, reverse], expected, rtol=2e-6, atol=2e-8)


def test_surface_and_prepared_projection_use_general_knot_bounds():
    """Single-start and prepared paths agree on nonnormalized edge/corner minima."""
    points = jnp.array(
        [[[0.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]]
    )
    surface = nrb.NurbsSurface(
        points,
        u_knots=jnp.array([2.0, 2.0, 4.0, 4.0]),
        v_knots=jnp.array([5.0, 5.0, 9.0, 9.0]),
    )
    queries = jnp.array([[0.3, -0.2, 1.2], [0.6, 0.6, 1.3], [1.0, 1.0, 1.0]])
    expected = np.array([[2.6, 2.0, 4.0], [7.4, 7.4, 9.0]])
    direct = surface.project_points(queries, gtol=1e-11)
    prepared = nrb.prepare_surface_projection(surface, samples_per_span=3)
    sparse = nrb.prepare_surface_projection(surface, samples_per_span=1)
    assert prepared.sample_parameters.shape[0] > sparse.sample_parameters.shape[0]
    assert prepared.sampled_points.shape[1] == prepared.sample_parameters.shape[0]
    assert_allclose(direct, expected, atol=1e-10)
    assert_allclose(prepared.project_points(queries, gtol=1e-11), expected, atol=1e-10)

    def project(query):
        return prepared.project_point(query, gtol=1e-11)

    expected_jacobian = np.array([[2.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    assert_allclose(jax.jit(jax.jacfwd(project))(queries[:, 0]), expected_jacobian, atol=1e-10)
    assert_allclose(jax.jit(jax.jacrev(project))(queries[:, 0]), expected_jacobian, atol=1e-10)
    with pytest.raises((TypeError, ValueError)):
        nrb.prepare_surface_projection(surface, samples_per_span=0)


def test_surface_projection_keeps_c0_seam_minimum():
    """A ridge minimum is stationary only along the free seam coordinate."""
    points = np.stack(
        (
            np.repeat(np.array([0.0, 1.0, 2.0])[:, None], 2, axis=1),
            np.repeat(np.array([0.0, 1.0, 0.0])[:, None], 2, axis=1),
            np.tile([0.0, 1.0], (3, 1)),
        )
    )
    surface = nrb.NurbsSurface(
        points,
        u_degree=1,
        v_degree=1,
        u_knots=jnp.array([0.0, 0.0, 0.5, 1.0, 1.0]),
    )
    queries = jnp.array([[1.0, 0.8], [1.5, 1.4], [0.4, 0.7]])
    expected = np.array([[0.5, 0.5], [0.4, 0.7]])
    assert_allclose(surface.project_points(queries, gtol=1e-10), expected, atol=1e-10)
    prepared = nrb.prepare_surface_projection(surface, samples_per_span=2)
    assert_allclose(prepared.project_points(queries, gtol=1e-10), expected, atol=1e-10)


@pytest.mark.parametrize("mode", [jax.jacfwd, jax.jacrev], ids=["forward", "reverse"])
def test_surface_projection_derivatives_respect_anisotropic_parameter_units(mode):
    """A regular plane remains regular when its parameter scales differ by 1e12."""
    points = jnp.array(
        [[[0.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]]
    )
    surface = nrb.NurbsSurface(
        points,
        u_knots=jnp.array([0.0, 0.0, 1e-6, 1e-6]),
        v_knots=jnp.array([0.0, 0.0, 1e6, 1e6]),
    )

    def project(point):
        return jnp.stack(surface.project_point_to_surface(point, gtol=1e-10))

    query = jnp.array([0.3, 0.6, 1.0])
    assert_allclose(project(query), [0.3e-6, 0.6e6], rtol=1e-10, atol=1e-15)
    expected = np.array([[1e-6, 0.0, 0.0], [0.0, 1e6, 0.0]])
    assert_allclose(jax.jit(mode(project))(query), expected, rtol=1e-10, atol=1e-15)


@pytest.mark.parametrize("mode", [jax.jacfwd, jax.jacrev], ids=["forward", "reverse"])
def test_integer_control_points_preserve_projection_and_query_derivatives(mode):
    """Integer geometry must not truncate the derivatives used by the solver."""
    points = np.array([[0, 1, 2], [0, 2, 0]])
    integer_curve = nrb.NurbsCurve(points)
    float_curve = nrb.NurbsCurve(points.astype(float))
    parameter = 0.31
    position = np.array([2 * parameter, 4 * parameter * (1 - parameter)])
    tangent = np.array([2.0, 4 - 8 * parameter])
    normal = np.array([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
    query = position + 0.1 * normal
    hessian = np.dot(tangent, tangent) - 8 * (position[1] - query[1])

    def project(point):
        return integer_curve.project_points(point[:, None], tol=1e-11)[0]

    assert_allclose(project(query), parameter, atol=1e-10)
    assert_allclose(project(query), float_curve.project_points(query[:, None])[0], atol=1e-10)
    assert_allclose(jax.jit(mode(project))(query), tangent / hessian, rtol=1e-8, atol=1e-10)


def test_weak_corner_stationary_maximum_is_not_accepted_as_projection():
    """Zero one-sided slopes still need nonnegative one-sided curvature."""
    curve = nrb.NurbsCurve(
        jnp.array([[0.0] * 5, [-10.0, 1.0, 1.0, 1.0, -10.0]]),
        degree=2,
        knots=jnp.array([0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0]),
    )
    query = jnp.zeros((2, 1))
    # With just one interior sample per span the nearest seed is the knot .5,
    # where both slopes vanish and the distance Hessian is negative on both sides.
    with pytest.raises(Exception, match="did not converge"):
        np.asarray(curve.project_points(query, samples_per_span=1, maxiter=0))
    parameter = curve.project_points(query)
    exact = 0.5 + np.array([-1.0, 1.0]) / (2 * np.sqrt(11))
    assert np.min(np.abs(np.asarray(parameter[0]) - exact)) < 1e-9
    assert_allclose(curve.get_value(parameter), np.zeros((2, 1)), atol=1e-9)


def test_skew_affine_surface_reaches_edge_without_clipping_newton_uphill():
    """A common feasible step fraction preserves descent for coupled parameters."""
    scale = np.sqrt(98.0)
    points = jnp.array(
        [[[0.0, -99.0], [1.0, -98.0]], [[0.0, scale], [0.0, scale]]]
    )
    surface = nrb.NurbsSurface(points)
    jacobian = np.array([[1.0, -99.0], [0.0, scale]])
    query = jacobian @ np.array([0.0625, 0.5625]) - np.array([0.02, 0.98 / scale])
    # At this value-grid seed H^-1 g=(1.01,.01); independently clipping the
    # coupled Newton components can turn the descent step into an uphill step.
    expected = np.array([0.0, 0.5625 - 5.1875 / 9899])
    actual = jnp.stack(surface.project_point_to_surface(query, gtol=1e-11))
    assert_allclose(actual, expected, rtol=0, atol=1e-10)
    prepared = nrb.prepare_surface_projection(surface)
    assert_allclose(prepared.project_point(query, gtol=1e-11), expected, rtol=0, atol=1e-10)


@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reversed"])
def test_compressed_rational_cubic_finds_polynomial_image_minimum(reverse):
    """Endpoint clusters expose a narrow basin missed by a uniform parameter grid."""
    points = np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 2.0, -2.0, 1.0]])
    weights = 1e4 ** np.arange(4)
    if reverse:
        points, weights = points[:, ::-1].copy(), weights[::-1].copy()
    curve = nrb.NurbsCurve(points, weights)
    queries = np.array([[0.0, 0.1, 0.1, 0.2], [0.0, 0.0, -0.1, 0.1]])
    # Geometric-progression weights only reparameterize the polynomial image:
    # s=R*u/(1+(R-1)*u). Enumerating its distance-derivative roots avoids a
    # reference grid that might itself miss the compressed rational basin.
    x = Polynomial([0.0, 0.0, 0.0, 1.0])
    y = Polynomial([1.0, 3.0, -15.0, 12.0])
    expected = []
    for qx, qy in queries.T:
        roots = ((x - qx) * x.deriv() + (y - qy) * y.deriv()).roots()
        interior = roots.real[
            (np.abs(roots.imag) < 1e-10) & (roots.real >= 0) & (roots.real <= 1)
        ]
        candidates = np.r_[0.0, interior, 1.0]
        expected.append(np.min((x(candidates) - qx) ** 2 + (y(candidates) - qy) ** 2))
    parameters = curve.project_points(queries)
    actual = jnp.sum((curve.get_value(parameters) - queries) ** 2, axis=0)
    assert_allclose(actual, expected, rtol=0, atol=1e-8)
