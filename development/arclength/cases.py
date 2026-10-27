"""Deterministic cases and independent SciPy/analytic reference calculations.

References intentionally do not call any nurbspy evaluation or production integration rule.
QUADPACK is split at every distinct knot, and warning messages are preserved.
Known analytic values override quadrature when an extreme rational layer could
also defeat the reference integrator.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import BSpline
from scipy.optimize import brentq


@dataclass
class Case:
    name: str
    P: np.ndarray
    W: np.ndarray
    degree: int
    knots: np.ndarray
    u1: float = 0.0
    u2: float = 1.0
    exact: float | None = None
    analytic_kind: str | None = None
    epsilon: float | None = None


def uniform_knots(n, degree):
    return np.r_[np.zeros(degree), np.linspace(0, 1, n - degree + 1), np.ones(degree)]


def get_cases():
    """Lengths: conventional shapes, knot jumps, scales, and rational stress."""
    line = np.array([[0., 3.], [0., 4.]])
    circle = np.array([[1., 1., 0.], [0., 1., 1.]])
    t = np.linspace(0, 1, 9)
    smooth = np.array([3 * t, np.sin(2.2 * np.pi * t), .3 * np.cos(3 * t)])
    repeat = np.array([[0., .3, .6, .8, 1., 1.5, 2., 2.6, 3.],
                       [0., .2, 1., -.5, 1., -1., .5, .6, 0.]])
    weights = np.array([1., .8, 1.2, .7, 1.4, .9, 1.1, .8, 1.])
    repeated_knots = np.r_[np.zeros(4), [.08, .3, .3, .3, .85], np.ones(4)]
    narrow_knots = np.r_[np.zeros(4), [.1, .10001, .10002, .7, .95], np.ones(4)]
    cases = [
        Case("line", line, np.ones(2), 1, uniform_knots(2, 1), exact=5., analytic_kind="line"),
        Case("quarter_circle", circle, np.array([1., 2**-.5, 1.]), 2,
             uniform_knots(3, 2), exact=np.pi/2, analytic_kind="circle"),
        Case("smooth_3d", smooth, weights, 3, uniform_knots(9, 3)),
        Case("repeated_knot", repeat, np.ones(9), 3, repeated_knots),
        Case("narrow_spans", repeat, np.ones(9), 3, narrow_knots),
        Case("collapsed", np.ones((2, 2)), np.ones(2), 1,
             uniform_knots(2, 1), exact=0., analytic_kind="collapsed"),
        Case("scale_1e-6", line*1e-6, np.ones(2), 1, uniform_knots(2, 1),
             exact=5e-6, analytic_kind="line"),
        Case("scale_1e6", line*1e6, np.ones(2), 1, uniform_knots(2, 1),
             exact=5e6, analytic_kind="line"),
        Case("rational_stress", smooth, np.array([1., .001, 1000., .01, 10., 1., .02, 200., 1.]),
             3, uniform_knots(9, 3)),
        Case("shifted_domain", smooth, weights, 3, 2.+3.*uniform_knots(9, 3), u1=2., u2=5.),
    ]
    for exponent in (6, 12, 16):
        cases.append(Case(f"weight_1e{exponent}", np.array([[0., 1.], [0., 0.]]),
                          np.array([1., 10.**exponent]), 1, uniform_knots(2, 1),
                          exact=1., analytic_kind="weighted_line"))
    cases.extend([
        Case("weight_1e12_reversed", np.array([[1., 0.], [0., 0.]]),
             np.array([1e12, 1.]), 1, uniform_knots(2, 1), exact=1., analytic_kind="weighted_line"),
        Case("weight_1e12_shifted", np.array([[0., 1.], [0., 0.]]),
             np.array([1., 1e12]), 1, 2.+3.*uniform_knots(2, 1), u1=2., u2=5.,
             exact=1., analytic_kind="weighted_line"),
    ])
    return cases + [replace(cases[2], name="smooth_partial", u1=.17, u2=.83)]


def get_mapping_cases():
    """Inverse tests distinguish zero speed from a nonunique flat interval."""
    conventional = {case.name: case for case in get_cases()}
    epsilon = 1e-4
    return [conventional[name] for name in
            ("line", "quarter_circle", "smooth_3d", "repeated_knot", "narrow_spans")] + [
        Case("quadratic_line", np.array([[0., .5, 2.], [0., 0., 0.]]),
             np.ones(3), 2, uniform_knots(3, 2), exact=2., analytic_kind="quadratic"),
        Case("near_zero_speed", np.array([[0., epsilon/3, 2*epsilon/3, 1.], [0., 0., 0., 0.]]),
             np.ones(4), 3, uniform_knots(4, 3), exact=1.,
             analytic_kind="near_zero", epsilon=epsilon),
        Case("endpoint_zero_speed", np.array([[0., 0., 0., 1.], [0., 0., 0., 0.]]),
             np.ones(4), 3, uniform_knots(4, 3), exact=1., analytic_kind="cubic"),
        Case("interior_zero_speed", np.array([[-1., 1., -1., 1.], [0., 0., 0., 0.]]),
             np.ones(4), 3, uniform_knots(4, 3), exact=2., analytic_kind="interior_zero"),
        Case("stationary_span", np.array([[0., 0., 1.], [0., 0., 0.]]), np.ones(3),
             1, np.array([0., 0., .4, 1., 1.]), exact=1., analytic_kind="stationary_span"),
        conventional["collapsed"],
    ]


def _reference_spline(case, P=None, W=None):
    P = case.P if P is None else np.asarray(P)
    W = case.W if W is None else np.asarray(W)
    return BSpline(case.knots, np.vstack((P*W, W)).T, case.degree)


def reference_speed(case, u, *, P=None, W=None):
    """Independent homogeneous quotient derivative in original coordinates."""
    spline = _reference_spline(case, P, W)
    value, derivative = spline(u), spline(u, nu=1)
    velocity = ((derivative[..., :-1]*value[..., -1, None]
                 - value[..., :-1]*derivative[..., -1, None]) / value[..., -1, None]**2)
    return np.linalg.norm(velocity, axis=-1)


def analytic_cumulative(case, u):
    """Known s(u) from the curve-domain start; None for nonanalytic cases."""
    u = np.asarray(u, dtype=float)
    a, b = case.knots[case.degree], case.knots[-case.degree-1]
    t = (u-a)/(b-a)
    kind = case.analytic_kind
    if kind in ("line", "collapsed"):
        return np.linalg.norm(case.P[:, -1]-case.P[:, 0])*t
    if kind == "weighted_line":
        ratio = case.W[-1]/case.W[0]
        return np.linalg.norm(case.P[:, -1]-case.P[:, 0]) * (ratio*t / ((1.-t)+ratio*t))
    if kind == "circle":
        return 2*np.arctan(t/(np.sqrt(2)+(1.-np.sqrt(2))*t))
    if kind == "quadratic":
        return t+t*t
    if kind == "near_zero":
        return case.epsilon*t+(1.-case.epsilon)*t**3
    if kind == "cubic":
        return t**3
    if kind == "interior_zero":
        return (2*t-1.)**3+1.
    if kind == "stationary_span":
        return np.maximum((t-.4)/.6, 0.)
    return None


def analytic_inverse(case, s):
    """Analytic/monotone scalar reference inverse, None for general curves.

    A stationary interval/collapsed curve has no unique inverse; this reference
    deliberately returns None for those cases instead of an arbitrary repair.
    """
    s = np.asarray(s, dtype=float)
    a, b = case.knots[case.degree], case.knots[-case.degree-1]
    kind = case.analytic_kind
    if kind == "line":
        t = s/case.exact
    elif kind == "weighted_line":
        q = s/case.exact
        ratio = case.W[-1]/case.W[0]
        t = q/(ratio*(1.-q)+q)
    elif kind == "circle":
        h = np.tan(s/2)
        t = np.sqrt(2)*h/(1.+(np.sqrt(2)-1.)*h)
    elif kind == "quadratic":
        t = 2*s/(1.+np.sqrt(1.+4*s))
    elif kind == "near_zero":
        t = np.array([brentq(lambda z: case.epsilon*z+(1.-case.epsilon)*z**3-q,
                            0., 1., xtol=5e-15) for q in s.ravel()]).reshape(s.shape)
    elif kind == "cubic":
        t = np.cbrt(s)
    elif kind == "interior_zero":
        t = (1.+np.cbrt(s-1.))/2
    else:
        return None
    return a+(b-a)*t


def reference_length(case, *, P=None, W=None, eps=2e-12, u1=None, u2=None):
    """Analytic or independent knot-span QUADPACK length with explicit warnings."""
    u1 = case.u1 if u1 is None else float(u1)
    u2 = case.u2 if u2 is None else float(u2)
    if P is None and W is None:
        start, end = analytic_cumulative(case, u1), analytic_cumulative(case, u2)
        if start is not None:
            return {"value": float(end-start), "error": 0., "evaluations": 0,
                    "analytic": True, "messages": [], "converged": True}
        if u1 == case.u1 and u2 == case.u2 and case.exact is not None:
            return {"value": case.exact, "error": 0., "evaluations": 0,
                    "analytic": True, "messages": [], "converged": True}
    lo, hi = sorted((u1, u2))
    bounds = np.unique(np.r_[lo, case.knots[(case.knots > lo) & (case.knots < hi)], hi])
    value, error, evaluations, messages = 0., 0., 0, []
    for left, right in zip(bounds[:-1], bounds[1:]):
        output = quad(lambda u: reference_speed(case, u, P=P, W=W), left, right,
                      epsabs=eps/max(1, len(bounds)-1), epsrel=eps,
                      limit=500, full_output=True)
        integral, estimate, info = output[:3]
        value += integral
        error += estimate
        evaluations += info["neval"]
        if len(output) > 3:
            messages.append(output[3])
    return {"value": value*(1 if u2 >= u1 else -1), "error": error,
            "evaluations": evaluations, "analytic": False,
            "messages": messages, "converged": not messages}


def reference_cumulative(case, u, *, eps=2e-12):
    """Vector partial-length reference for full inverse round-trip validation."""
    u = np.asarray(u, dtype=float)
    exact = analytic_cumulative(case, u)
    if exact is not None:
        return exact
    a = case.knots[case.degree]
    return np.array([reference_length(case, u1=a, u2=t, eps=eps)["value"]
                     for t in u.ravel()]).reshape(u.shape)
