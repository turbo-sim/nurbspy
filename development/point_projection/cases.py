"""Deterministic geometry/query cases and independent SciPy references."""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import brentq, minimize_scalar


@dataclass
class Case:
    name: str
    kind: str
    points: np.ndarray
    weights: np.ndarray
    knots_u: np.ndarray
    degree_u: int
    queries: np.ndarray
    knots_v: np.ndarray | None = None
    degree_v: int | None = None
    note: str = ""


def open_knots(count, degree, internal=None):
    if internal is None:
        internal = np.linspace(0.0, 1.0, count - degree + 1)[1:-1]
    return np.r_[np.zeros(degree + 1), internal, np.ones(degree + 1)]


def curve_cases():
    rng = np.random.default_rng(27)
    queries = np.vstack((np.linspace(-0.2, 1.2, 16), rng.uniform(-0.4, 0.8, 16)))
    line = Case("line_endpoints", "curve", np.array([[0., 1.], [0., 0.]]), np.ones(2),
                open_knots(2, 1), 1, queries, note="Exact clipped-line projection, including exterior queries.")
    arc = Case("rational_arc", "curve", np.array([[1., 1., 0.], [0., 1., 1.]]),
               np.array([1., np.sqrt(0.5), 1.]), open_knots(3, 2), 2,
               np.array([[1.2, .8, -.2, 1.4, .3, 0., .9, .2], [.4, .9, 1.2, -.2, .5, 0., .1, .9]]),
               note="Exact rational quarter-circle; center query has nonunique projection.")
    P = np.array([[0., .6, .2, .9, .3, 1.1, .5, 1.], [0., 1.4, -1.2, 1.1, -1.3, 1.2, -.9, .2]])
    folded = Case("folded_nonuniform", "curve", P, np.ones(8),
                  open_knots(8, 3, [.04, .18, .66, .93]), 3,
                  np.vstack((rng.uniform(-.2, 1.2, 16), rng.uniform(-.7, .9, 16))),
                  note="Several competing local minima and strongly nonuniform knot spans.")
    weighted = Case("high_weight", "curve", P[:, :6], np.array([1., .02, 80., .04, 25., 1.]),
                    open_knots(6, 3, [.12, .85]), 3,
                    np.vstack((rng.uniform(-.2, 1.2, 16), rng.uniform(-1., 1., 16))),
                    note="Positive rational weights spanning 4000:1; concentrated geometric variation.")
    kink = Case("polyline_corner", "curve", np.array([[0., 1., 2.], [0., 1., 0.]]), np.ones(3),
                np.array([0., 0., .5, 1., 1.]), 1,
                np.array([[1., .8, 1.2, -.2, 2.2], [1.5, 1.4, 1.4, -.2, -.2]]),
                note="C0 interior-knot minima need one-sided conditions, not smooth stationarity.")
    nonclamped = Case("nonclamped_domain", "curve", np.array([[0., 1., 2., 3., 4.], [0., 1., -.5, 1., 0.]]),
                     np.ones(5), np.arange(8, dtype=float), 2,
                     np.array([[.2, 1.5, 3.8, 2.5], [.3, -.1, .4, .2]]),
                     note="Active domain [2,5] with nonclamped basis functions.")
    maximum = Case("two_minima", "curve", np.array([[0., .5, 1.], [0., 2., 0.]]), np.ones(3),
                   open_knots(3, 2), 2, np.array([[.5, .5, .4, .6], [.1, -.2, .1, .1]]),
                   note="Symmetric competing minima and a stationary maximum at u=0.5 for low queries.")
    degenerate = Case("collapsed", "curve", np.zeros((2, 3)), np.ones(3), open_knots(3, 2), 2,
                      np.array([[1., 0.], [.2, 0.]]), note="Every parameter is a minimizer; forward values exist, implicit derivatives do not.")
    return {case.name: case for case in [line, arc, folded, weighted, kink, nonclamped, maximum, degenerate]}


def surface_cases():
    plane = np.array([[[0., 0.], [1., 1.]], [[0., 1.], [0., 1.]], [[0., 0.], [0., 0.]]])
    queries = np.array([[.3, -.2, 1.2, .5, -.4, .8], [.6, .6, 1.3, -.2, -.3, .7], [1., 1., 1., -.3, .2, -.4]])
    a = Case("plane_bounds", "surface", plane, np.ones((2, 2)), open_knots(2, 1), 1, queries,
             open_knots(2, 1), 1, "Analytic interior, edge and corner projections.")
    # Extruding a folded planar curve produces a surface with competing u basins
    # while leaving v exactly identifiable. Its reference separates into a
    # curve projection and an analytic bounded extrusion coordinate.
    curve = curve_cases()["folded_nonuniform"]
    points = np.stack((np.repeat(curve.points[0, :, None], 2, axis=1),
                       np.repeat(curve.points[1, :, None], 2, axis=1),
                       np.tile([0., 1.], (curve.points.shape[1], 1))))
    c = Case("folded_extrusion", "surface", points, np.ones(points.shape[1:]), curve.knots_u, 3,
             np.vstack((curve.queries[:, :8], np.linspace(-.2, 1.2, 8))), open_knots(2, 1), 1,
             "Multiple u basins with exactly bounded extrusion coordinate v.")
    d = Case("plane_general_domain", "surface", plane, np.ones((2, 2)), np.array([2., 2., 4., 4.]), 1,
             queries[:, :3], np.array([5., 5., 9., 9.]), 1,
             "Plane reparameterized onto the active domain [2,4]x[5,9].")
    return {case.name: case for case in [a, c, d]}


def numpy_evaluator(case):
    """Return rational values/first/second derivatives using SciPy B-splines."""
    bu = BSpline(case.knots_u, np.eye(case.points.shape[1]), case.degree_u)
    bv = None if case.kind == "curve" else BSpline(case.knots_v, np.eye(case.points.shape[2]), case.degree_v)
    weighted = case.points * case.weights

    def basis(spline, parameter, order):
        if order > spline.k:
            return np.zeros_like(spline(parameter))
        return spline(parameter, nu=order)

    def evaluate(parameters):
        parameters = np.atleast_1d(parameters)
        if case.kind == "curve":
            rows = [basis(bu, parameters[0], order) for order in range(3)]
        else:
            rows = [np.outer(basis(bu, parameters[0], i), basis(bv, parameters[1], j))
                    for i, j in [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]]
        numerator = [np.sum(weighted * row, axis=tuple(range(1, weighted.ndim))) for row in rows]
        denominator = [np.sum(case.weights * row) for row in rows]
        C = numerator[0] / denominator[0]
        if case.kind == "curve":
            dC = (numerator[1] - C * denominator[1]) / denominator[0]
            ddC = (numerator[2] - 2 * dC * denominator[1] - C * denominator[2]) / denominator[0]
            return C, dC[:, None], ddC[:, None, None]
        du = (numerator[1] - C * denominator[1]) / denominator[0]
        dv = (numerator[2] - C * denominator[2]) / denominator[0]
        duu = (numerator[3] - 2 * du * denominator[1] - C * denominator[3]) / denominator[0]
        duv = (numerator[4] - du * denominator[2] - dv * denominator[1] - C * denominator[4]) / denominator[0]
        dvv = (numerator[5] - 2 * dv * denominator[2] - C * denominator[5]) / denominator[0]
        return C, np.stack((du, dv), axis=1), np.stack((np.stack((duu, duv), axis=1), np.stack((duv, dvv), axis=1)), axis=1)
    return evaluate


def independent_reference(case):
    """Independent SciPy/analytic distance reference for these fixed fixtures.

    Curve reference candidates are active knots and stationary roots bracketed
    on dense span grids (with endpoint clusters). Scalar minimization polishes
    sampled minima when no smooth sign bracket exists. This remains a numerical
    reference, not a global certificate for arbitrary rational geometry.

    Surface fixtures have separable analytic structure, so no competing local
    surface solves or alternative projection implementation are maintained.
    """
    if case.kind == "surface":
        if case.name == "folded_extrusion":
            curve = Case(case.name, "curve", case.points[:2, :, 0],
                         case.weights[:, 0], case.knots_u, case.degree_u,
                         case.queries[:2])
            u, distances = independent_reference(curve)
            v = np.clip(case.queries[2], 0., 1.)
            return np.column_stack((u[:, 0], v)), distances + (v - case.queries[2])**2
        # The two plane fixtures share the unit square geometry, with different
        # affine parameter domains. Their bounded geometric solution is exact.
        xy = np.clip(case.queries[:2], 0., 1.)
        lower = np.array([case.knots_u[case.degree_u], case.knots_v[case.degree_v]])
        upper = np.array([case.knots_u[-case.degree_u-1], case.knots_v[-case.degree_v-1]])
        parameters = lower[:, None] + (upper-lower)[:, None] * xy
        distances = np.sum((xy-case.queries[:2])**2, axis=0) + case.queries[2]**2
        return parameters.T, distances

    evaluate = numpy_evaluator(case)
    lower, upper = case.knots_u[case.degree_u], case.knots_u[-case.degree_u-1]
    boundaries = np.unique(case.knots_u[(case.knots_u >= lower) & (case.knots_u <= upper)])
    knots, multiplicity = np.unique(case.knots_u, return_counts=True)
    nonsmooth = knots[(knots > lower) & (knots < upper) & (multiplicity >= case.degree_u)]
    samples = [boundaries]
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        samples.extend([np.linspace(a, b, 4097),
                        a+(b-a)*10.**np.linspace(-10, -1, 40),
                        b-(b-a)*10.**np.linspace(-10, -1, 40)])
    grid = np.unique(np.concatenate(samples))
    basis = BSpline(case.knots_u, np.eye(case.points.shape[1]), case.degree_u)(grid).T
    positions = (case.points*case.weights) @ basis / (case.weights @ basis)[None]
    parameters, distances = [], []
    for point in case.queries.T:
        def objective(u):
            position, jacobian, _ = evaluate([u])
            residual = position-point
            return float(residual @ residual), float(2*jacobian[:, 0] @ residual)

        costs = np.sum((positions-point[:, None])**2, axis=0)
        candidates = [(u, objective(u)[0]) for u in boundaries]
        variable = np.ptp(costs) > 128*np.finfo(float).eps*max(1., np.max(np.abs(costs)))
        minima = np.flatnonzero((costs[1:-1] <= costs[:-2]) & (costs[1:-1] <= costs[2:]))+1
        for i in minima if variable else []:
            a, b = grid[i-1], grid[i+1]
            result = minimize_scalar(lambda u: objective(u)[0], bounds=(a, b),
                                     method="bounded", options={"xatol": 1e-14})
            candidates.extend([(grid[i], costs[i]), (result.x, result.fun)])
            smooth = not np.any((nonsmooth > a) & (nonsmooth < b))
            if smooth and objective(a)[1] <= 0 <= objective(b)[1]:
                u = brentq(lambda value: objective(value)[1], a, b,
                           xtol=5e-15, rtol=1e-14)
                candidates.append((u, objective(u)[0]))
        u, value = min(candidates, key=lambda pair: pair[1])
        parameters.append([u])
        distances.append(value)
    return np.asarray(parameters), np.asarray(distances)
