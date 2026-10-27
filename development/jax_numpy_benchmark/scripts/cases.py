"""Deterministic geometry; no random state, test-suite imports or timed setup."""
from functools import lru_cache
import numpy as np


def curve_data(kind="rational", count=257):
    p = 3
    P = np.array([[0, .3, .8, 1.2, 1.7, 2], [0, .5, -.2, .8, .3, .1],
                  [0, .1, .4, -.1, .6, .2]], dtype=float)
    W = np.array([1, .8, 1.4, .9, 1.2, 1.])
    U = np.array([0, 0, 0, 0, .3, .7, 1, 1, 1, 1.])
    if kind == "polynomial":
        W = np.ones_like(W)
    elif kind == "planar":
        P = P[:2]
    elif kind == "evolution":
        P = P[:1]
    elif kind == "repeated":
        U[4:6] = .5
    elif kind == "circle":
        p, P, W, U = 2, np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0.]]), np.array([1, 2**-.5, 1]), np.array([0, 0, 0, 1, 1, 1.])
    elif kind == "line":
        p, P, W, U = 1, np.array([[0, 2], [0, 0], [0, 0.]]), np.ones(2), np.array([0, 0, 1, 1.])
    elif kind == "collapsed":
        P[:] = 0
    elif kind == "quintic":
        p, U = 5, np.r_[np.zeros(6), np.ones(6)]
    elif kind == "constant":
        p, P, W, U = 0, P[:, :1], np.ones(1), np.array([0., 1.])
    elif kind == "weight_stress":
        W[0] = 1e-6
    u = np.linspace(0, 1, count)
    if kind == "shifted":
        U, u = -2+5*U, -2+5*u
    elif kind == "nonclamped":
        U = np.arange(P.shape[1]+p+1, dtype=float)
        u = np.linspace(U[p], U[P.shape[1]], count)
        # Highest derivatives are one-sided at this boundary; use the left
        # limit to match SciPy's end-of-active-domain convention.
        u[-1] = np.nextafter(u[-1], u[0])
    return P, W, U, u, p


def surface_data(kind="rational", count=17):
    a, b = np.meshgrid(np.linspace(0, 1, 6 if kind == "repeated" else 4), np.linspace(0, 1, 4), indexing="ij")
    P = np.stack((2*a, b, .4*np.sin(np.pi*a)*np.sin(np.pi*b)))
    W = 1 + .25*np.cos(2*a+3*b)
    if kind == "polynomial":
        W = np.ones_like(W)
    p = q = 3
    U = V = np.array([0, 0, 0, 0, 1, 1, 1, 1.])
    if kind == "repeated":
        U = np.array([0, 0, 0, 0, .5, .5, 1, 1, 1, 1.])
    if kind in ("plane", "paraboloid"):
        a, b = np.meshgrid([0, .5, 1.], [0, .5, 1.], indexing="ij")
        P = np.stack((a, b, np.zeros_like(a)))
        if kind == "paraboloid":
            sq = np.array([0, 0, 1.])
            P[2] = sq[:, None] + sq[None, :]
        W, p, q = np.ones((3, 3)), 2, 2
        U = V = np.array([0, 0, 0, 1, 1, 1.])
    u, v = np.meshgrid(np.linspace(0, 1, count), np.linspace(0, 1, count), indexing="ij")
    return P, W, U, V, u.ravel(), v.ravel(), p, q


def generator(backend, kind, xp, scale):
    """Same rational boundary data for both production constructors."""
    P = scale*xp.asarray([[1, 1.3, 1.1], [0, .2, 0], [0, .4, 1.]])
    W = xp.asarray([1, .8, 1.])
    c = backend.NurbsCurve(P, W)
    if kind == "bilinear":
        corners = [xp.asarray(x)*scale for x in ([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, .3])]
        return backend.NurbsSurfaceBilinear(*corners).NurbsSurface
    if kind == "ruled":
        other = backend.NurbsCurve(P+xp.asarray([0, .4, 1.1])[:, None], W)
        return backend.NurbsSurfaceRuled(c, other).NurbsSurface
    if kind == "extruded":
        return backend.NurbsSurfaceExtruded(c, xp.asarray([.4, .2, 1.]), 1.1).NurbsSurface
    if kind == "revolution":
        options = {"n_arcs": 1} if backend.__name__.endswith(".jax") else {}
        return backend.NurbsSurfaceRevolution(c, xp.asarray([.04, 0, 0]), xp.asarray([.2, .1, 1.]), .1, 1.1, **options).NurbsSurface
    south = scale*xp.asarray([[0, .5, 1], [0, 0, 0], [0, 0, 0.]])
    north = south+xp.asarray([0, 1., 0])[:, None]
    west = xp.stack((south[:, 0], xp.asarray([0, .5, .4]), north[:, 0]), axis=1)
    east = xp.stack((south[:, -1], xp.stack((scale, xp.asarray(.5), xp.asarray(1.1))), north[:, -1]), axis=1)
    return backend.NurbsSurfaceCoons(*[backend.NurbsCurve(x, W) for x in (south, north, west, east)]).NurbsSurface


@lru_cache(maxsize=None)
def gauss(order=16):
    """Reuse geometry-independent nodes for both backends, not just XLA."""
    x, w = np.polynomial.legendre.leggauss(order)
    x, w = (x+1)/2, w/2
    x.setflags(write=False)
    w.setflags(write=False)
    return x, w


@lru_cache(maxsize=None)
def gauss_grid(order=16):
    x, w = gauss(order)
    u, v = np.meshgrid(x, x, indexing="ij")
    result = (u.ravel(), v.ravel(), np.outer(w, w).ravel())
    for array in result:
        array.setflags(write=False)
    return result
