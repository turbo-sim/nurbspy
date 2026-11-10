"""Local projection, sparse basis derivatives and constrained fitting geometry."""

from time import perf_counter
import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import coo_matrix, csr_matrix, vstack
from nurbspy.fitting._common import basis_table
from _experiment import make_surface, target


def paired_basis(initial, uv, *, derivatives=False):
    """Reuse the generic local support tables; add custom-constraint derivatives."""
    n, m = initial.W.shape
    support, values = basis_table(initial, uv)
    width = support.shape[1]
    indices = support.ravel()
    indptr = np.arange(0, width * (len(uv) + 1), width)

    def matrix(values):
        return csr_matrix((values.ravel(), indices, indptr), shape=(len(uv), n * m))

    value = matrix(values)
    if not derivatives:
        return value
    rows = np.arange(len(uv))[:, None]
    iu, iv = support // m, support % m
    a = BSpline(initial.U, np.eye(n), initial.p)(uv[:, 0])[rows, iu]
    b = BSpline(initial.V, np.eye(m), initial.q)(uv[:, 1])[rows, iv]
    da = BSpline(initial.U, np.eye(n), initial.p)(uv[:, 0], nu=1)[rows, iu]
    db = BSpline(initial.V, np.eye(m), initial.q)(uv[:, 1], nu=1)[rows, iv]
    return value, matrix(da * b), matrix(a * db)


def terms(b, c, w, bu=None, bv=None):
    sample = np.repeat(np.arange(b.shape[0]), np.diff(b.indptr))
    denominator = b @ w
    s = (b @ (c * w[:, None])) / denominator[:, None]
    r = b.data * w[b.indices] / denominator[sample]
    result = dict(s=s, r=r, sample=sample, indices=b.indices)
    if bu is not None:
        du = (bu @ w) / denominator
        dv = (bv @ w) / denominator
        su = (bu @ (c * w[:, None])) / denominator[:, None] - s * du[:, None]
        sv = (bv @ (c * w[:, None])) / denominator[:, None] - s * dv[:, None]
        ru = bu.data * w[b.indices] / denominator[sample] - r * du[sample]
        rv = bv.data * w[b.indices] / denominator[sample] - r * dv[sample]
        result.update(su=su, sv=sv, ru=ru, rv=rv)
    return result


def geometry_jacobian(state, c):
    """Sample-major residual Jacobian for CP coordinates and log W[1:]."""
    sample, idx, r = state["sample"], state["indices"], state["r"]
    rows = (sample[:, None] * 3 + np.arange(3)).ravel()
    cols = (idx[:, None] * 3 + np.arange(3)).ravel()
    free = np.repeat(idx > 0, 3)
    wvals = (r[:, None] * (c[idx] - state["s"][sample])).ravel()[free]
    return coo_matrix(
        (
            np.r_[np.repeat(r, 3), wvals],
            (
                np.r_[rows, rows[free]],
                np.r_[cols, np.repeat(3 * len(c) + idx - 1, 3)[free]],
            ),
        ),
        shape=(3 * len(state["s"]), 4 * len(c) - 1),
    ).tocsr()


def area_quadrature(initial, order=6):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    intervals = np.unique(initial.U)
    axis = np.concatenate(
        [
            (a + b) / 2 + (b - a) / 2 * nodes
            for a, b in zip(intervals[:-1], intervals[1:])
        ]
    )
    wa = np.concatenate(
        [(b - a) / 2 * weights for a, b in zip(intervals[:-1], intervals[1:])]
    )
    u, v = np.meshgrid(axis, axis, indexing="ij")
    return np.stack((u.ravel(), v.ravel()), axis=1), np.outer(wa, wa).ravel()


def analytic_area(order=100):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    u, v = np.meshgrid((nodes + 1) / 2, (nodes + 1) / 2, indexing="ij")
    du = np.zeros_like(u)
    dv = np.zeros_like(u)
    # Independent quadrature of the analytic graph; no fitted reference surface.
    for height, x, y, sx, sy in (
        (0.48, 0.23, 0.25, 0.12, 0.14),
        (0.42, 0.75, 0.67, 0.14, 0.12),
        (0.34, 0.25, 0.82, 0.12, 0.11),
        (-0.38, 0.55, 0.30, 0.13, 0.13),
        (-0.32, 0.50, 0.78, 0.11, 0.12),
        (-0.28, 0.85, 0.27, 0.10, 0.13),
    ):
        term = height * np.exp(-0.5 * (((u - x) / sx) ** 2 + ((v - y) / sy) ** 2))
        du -= term * (u - x) / sx**2
        dv -= term * (v - y) / sy**2
    return float(np.sum(np.outer(weights, weights) / 4 * np.sqrt(1 + du**2 + dv**2)))


def constrained_problem(
    initial, observations, *, area_fraction=0.85, quadrature_order=6
):
    """Fixed planar CPs, unit edge weights, and area/orientation inequalities."""
    count = initial.W.size
    interior = np.ones(initial.W.shape, dtype=bool)
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    free = np.flatnonzero(interior.ravel())
    columns = np.r_[3 * np.arange(count) + 2, 3 * count + free - 1]
    weights_columns = np.full(count, -1, dtype=int)
    weights_columns[free] = count + np.arange(len(free))
    c0 = initial.P.reshape(3, -1).T.copy()
    u, v = np.meshgrid(observations["axis"], observations["axis"], indexing="ij")
    training = paired_basis(initial, np.stack((u.ravel(), v.ravel()), axis=1))
    area_uv, omega = area_quadrature(
        initial, max(quadrature_order, int(np.ceil(20 / (initial.W.shape[0] - 3))))
    )
    area_matrices = paired_basis(initial, area_uv, derivatives=True)
    guard = np.linspace(0, 1, max(9, 2 * initial.W.shape[0] + 1))
    gu, gv = np.meshgrid(guard, guard, indexing="ij")
    guard_matrices = paired_basis(
        initial, np.stack((gu.ravel(), gv.ravel()), axis=1), derivatives=True
    )
    reference_area = analytic_area()
    limit = 1 + area_fraction * (reference_area - 1)
    scale = np.sqrt(len(observations["points"]))
    cache = {}

    def unpack(x):
        c = c0.copy()
        c[:, 2] = x[:count]
        w = np.ones(count)
        w[free] = np.exp(x[count:])
        return c, w

    def geometry(x):
        return make_surface(initial, *unpack(x))

    def values(x):
        if "x" in cache and np.array_equal(x, cache["x"]):
            return cache
        c, w = unpack(x)
        data = terms(training, c, w)
        residual = ((data["s"] - observations["points"]) / scale).ravel()
        jac = geometry_jacobian(data, c)[:, columns] / scale
        a = terms(*area_matrices[:1], c, w, *area_matrices[1:])
        cross = np.cross(a["su"], a["sv"])
        length = np.linalg.norm(cross, axis=1)
        normal = cross / np.maximum(length, 1e-15)[:, None]
        grad_u = np.cross(a["sv"], normal)
        grad_v = np.cross(normal, a["su"])
        sample, idx = a["sample"], a["indices"]
        cp_gradient = np.bincount(
            idx,
            weights=omega[sample]
            * (a["ru"] * grad_u[sample, 2] + a["rv"] * grad_v[sample, 2]),
            minlength=count,
        )
        difference = c[idx] - a["s"][sample]
        dsu = a["ru"][:, None] * difference - a["r"][:, None] * a["su"][sample]
        dsv = a["rv"][:, None] * difference - a["r"][:, None] * a["sv"][sample]
        weight_gradient = np.bincount(
            idx,
            weights=omega[sample]
            * np.sum(grad_u[sample] * dsu + grad_v[sample] * dsv, axis=1),
            minlength=count,
        )
        area = float(omega @ length)
        area_gradient = np.r_[cp_gradient, weight_gradient[free]] / limit
        g = terms(*guard_matrices[:1], c, w, *guard_matrices[1:])
        determinant = g["su"][:, 0] * g["sv"][:, 1] - g["su"][:, 1] * g["sv"][:, 0]
        guard_u = np.stack(
            (g["sv"][:, 1], -g["sv"][:, 0], np.zeros(len(determinant))), axis=1
        )
        guard_v = np.stack(
            (-g["su"][:, 1], g["su"][:, 0], np.zeros(len(determinant))), axis=1
        )
        sample, idx = g["sample"], g["indices"]
        diff = c[idx] - g["s"][sample]
        dsu = g["ru"][:, None] * diff - g["r"][:, None] * g["su"][sample]
        dsv = g["rv"][:, None] * diff - g["r"][:, None] * g["sv"][sample]
        entries = -np.sum(guard_u[sample] * dsu + guard_v[sample] * dsv, axis=1)
        valid = weights_columns[idx] >= 0
        guard_jac = coo_matrix(
            (entries[valid], (sample[valid], weights_columns[idx][valid])),
            shape=(len(determinant), len(x)),
        ).tocsr()
        constraints = np.r_[area / limit - 1, 0.05 - determinant]
        constraint_jac = vstack(
            (csr_matrix(area_gradient[None]), guard_jac), format="csr"
        )
        cache.update(
            x=x.copy(),
            residual=residual,
            jac=jac,
            g=constraints,
            gjac=constraint_jac,
            area=area,
            minimum_xy_jacobian=float(determinant.min()),
        )
        return cache

    x0 = np.zeros(count + len(free))
    bounds = (
        np.r_[np.full(count, -0.6), np.full(len(free), np.log(0.5))],
        np.r_[np.full(count, 0.6), np.full(len(free), np.log(2.0))],
    )
    return (
        x0,
        bounds,
        geometry,
        values,
        dict(
            reference_area=reference_area,
            area_limit=limit,
            area_fraction=area_fraction,
            variables=len(x0),
            fixed_planar_coordinates=True,
            boundary_weights_fixed=True,
            height_bounds=[-0.6, 0.6],
            weight_bounds=[0.5, 2.0],
        ),
    )


def project_distances(model, points):
    """Local one-sided distances from NurbsPy's existing public projection API."""
    from _projection import project_points

    return project_points(model, points)[1]
