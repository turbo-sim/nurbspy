"""Analytic stress cases for value-grid initialization followed by one solve.

The hidden-basin pair has exactly the same cubic Bezier image as an ordinary
polynomial curve. Geometric-progression rational weights only reparameterize
it: s = R*u/(1 + (R-1)*u). Most of its image lies between the first few uniform
u samples. The origin-facing endpoint is a strict local minimum, so choosing
that endpoint from a coarse grid can trap a perfectly valid bounded solve.

These cases probe sampling and conditioning limits. All weights are positive
and the geometry is finite and regular; this alone does not make a uniform
parameter grid sufficient.
"""

import numpy as np
from numpy.polynomial import Polynomial

from cases import Case, independent_reference, open_knots


def extra_cases():
    """Return three deterministic positive-weight conditioning stress cases."""
    line = Case(
        "weighted_line_layer", "curve", np.array([[0., 1.], [0., 0.]]),
        np.array([1., 1e8]), open_knots(2, 1), 1,
        np.array([[.25, .5, .75, .99], [.1, -.2, .1, .2]]),
        note="Exact unit line; the u*=~99/R query probes an unresolved endpoint speed layer.",
    )
    points = np.array([[0., 0., 0., 1.], [1., 2., -2., 1.]])
    queries = np.array([[0., .1, .1, .2], [0., 0., -.1, .1]])
    ratio = 1e4
    hidden = Case(
        "hidden_basin_reparam", "curve", points, ratio ** np.arange(4),
        open_knots(4, 3), 3, queries,
        note="Cubic Bezier with weights R**i, R=1e4. Its correct interior basin is compressed near u=0; u=0 is another strict local minimum.",
    )
    mirror = Case(
        "hidden_basin_mirror", "curve", points[:, ::-1].copy(),
        hidden.weights[::-1].copy(), open_knots(4, 3), 3, queries.copy(),
        note="Same curve image and queries, reversed parameterization; compressed basin lies near u=1.",
    )
    return {case.name: case for case in (line, hidden, mirror)}


def _polynomial_reference(queries):
    """Enumerate every real stationary root of the degree-five distance derivative.

    For the unweighted cubic, x(s)=s**3 and y(s)=1+3*s-15*s**2+12*s**3.
    Polynomial roots plus both endpoints give an independent finite candidate
    set. Unlike a dense original-u reference, this does not miss the compressed
    rational parameter layer. This is ordinary float64 algebra, not certified
    root isolation or a proof for arbitrary NURBS.
    """
    x = Polynomial([0., 0., 0., 1.])
    y = Polynomial([1., 3., -15., 12.])
    parameters, distances = [], []
    for qx, qy in queries.T:
        roots = ((x - qx) * x.deriv() + (y - qy) * y.deriv()).roots()
        real = roots.real[(np.abs(roots.imag) < 1e-10) & (roots.real >= 0) & (roots.real <= 1)]
        candidates = np.r_[0., real, 1.]
        costs = (x(candidates) - qx) ** 2 + (y(candidates) - qy) ** 2
        best = np.argmin(costs)
        parameters.append(candidates[best])
        distances.append(costs[best])
    return np.asarray(parameters), np.asarray(distances)


def extra_reference(case):
    """Return independent parameter rows and squared distances for an extra case.

    Unknown cases delegate to the independent SciPy reference, so callers
    can use this entry point for either collection without name dispatch.
    """
    if case.name == "weighted_line_layer":
        qx, qy = case.queries
        position = np.clip(qx, 0., 1.)
        ratio = case.weights[1] / case.weights[0]
        parameters = position / (ratio * (1 - position) + position)
        return parameters[:, None], (qx - position) ** 2 + qy ** 2
    if case.name in {"hidden_basin_reparam", "hidden_basin_mirror"}:
        base, distances = _polynomial_reference(case.queries)
        ratio = 1e4
        parameters = base / (ratio * (1 - base) + base)
        if case.name.endswith("mirror"):
            parameters = 1 - parameters
        return parameters[:, None], distances
    return independent_reference(case)
