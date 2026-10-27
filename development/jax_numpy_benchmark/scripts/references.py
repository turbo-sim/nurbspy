"""Independent SciPy BSpline evaluations and rational quotient recurrences.

These references deliberately do not call nurbspy. Projection references are
multi-start numerical searches, not certificates of global optimality.
"""
from math import comb
import numpy as np
from scipy.interpolate import BSpline
from scipy.integrate import quad
from scipy.optimize import minimize, minimize_scalar


def basis(U, p, u, order=0):
    n = len(U)-p-1
    if order > p:
        return np.zeros((n, np.size(u)))
    return BSpline(U, np.eye(n), p)(np.atleast_1d(u), nu=order).T


class Curve:
    def __init__(self, P, W, U, p):
        self.P, self.W, self.U, self.p = P, W, U, p

    def get_derivative(self, u, order):
        H = np.concatenate((self.P*self.W, self.W[None]))
        d = [H@basis(self.U, self.p, u, k) for k in range(order+1)]
        out = []
        for k in range(order+1):
            value = d[k][:-1].copy()
            for j in range(1, k+1):
                value -= comb(k, j)*d[j][-1]*out[k-j]
            out.append(value/d[0][-1])
        return out[-1]

    def get_value(self, u):
        return self.get_derivative(u, 0)

    def get_tangent(self, u):
        d = self.get_derivative(u, 1)
        return d/np.linalg.norm(d, axis=0)

    def get_normal(self, u):
        a, b = self.get_derivative(u, 1), self.get_derivative(u, 2)
        if a.shape[0] == 2:
            a, b = np.vstack((a, np.zeros(a.shape[1]))), np.vstack((b, np.zeros(b.shape[1])))
        n = np.cross(np.cross(a.T, b.T), a.T).T
        return (n/np.linalg.norm(n, axis=0))[:self.P.shape[0]]

    def get_binormal(self, u):
        a, b = self.get_derivative(u, 1), self.get_derivative(u, 2)
        if a.shape[0] == 2:
            a, b = np.vstack((a, np.zeros(a.shape[1]))), np.vstack((b, np.zeros(b.shape[1])))
        n = np.cross(a.T, b.T).T
        return (n/np.linalg.norm(n, axis=0))[:self.P.shape[0]]

    def get_curvature(self, u):
        a, b = self.get_derivative(u, 1), self.get_derivative(u, 2)
        cross = a[0]*b[1]-a[1]*b[0] if a.shape[0] == 2 else np.cross(a.T, b.T)
        size = np.abs(cross) if a.shape[0] == 2 else np.linalg.norm(cross, axis=-1)
        return size/np.linalg.norm(a, axis=0)**3

    def get_torsion(self, u):
        a, b, c = [self.get_derivative(u, k) for k in (1, 2, 3)]
        cross = np.cross(a.T, b.T).T
        return np.sum(cross*c, axis=0)/np.sum(cross*cross, axis=0)

    def get_arclength(self, u1=0, u2=1):
        cuts = np.unique(np.r_[u1, self.U[(self.U > u1) & (self.U < u2)], u2])
        return sum(quad(lambda x: float(np.linalg.norm(self.get_derivative(x, 1))), a, b, epsabs=1e-11, epsrel=1e-11)[0] for a, b in zip(cuts[:-1], cuts[1:]))

    def project(self, points):
        results = []
        cuts = np.unique(np.r_[self.U, np.linspace(0, 1, 65)])
        for point in points:
            fun = lambda u: float(np.sum((self.get_value(u)[:, 0]-point)**2))
            candidates = list(cuts)
            candidates += [minimize_scalar(fun, bounds=(a, b), method="bounded", options={"xatol": 1e-14}).x for a, b in zip(cuts[:-1], cuts[1:])]
            results.append(min(candidates, key=fun))
        return np.asarray(results)


class Surface:
    def __init__(self, P, W, U, V, p, q):
        self.P, self.W, self.U, self.V, self.p, self.q = P, W, U, V, p, q

    def get_derivative(self, u, v, order_u, order_v):
        H = np.concatenate((self.P*self.W, self.W[None]))
        raw, out = {}, {}
        for i in range(order_u+1):
            for j in range(order_v+1):
                raw[i, j] = np.einsum("dij,ik,jk->dk", H, basis(self.U, self.p, u, i), basis(self.V, self.q, v, j))
                val = raw[i, j][:-1].copy()
                for a in range(i+1):
                    for b in range(j+1):
                        if a or b:
                            val -= comb(i, a)*comb(j, b)*raw[a, b][-1]*out[i-a, j-b]
                out[i, j] = val/raw[0, 0][-1]
        return out[order_u, order_v]

    def get_value(self, u, v):
        return self.get_derivative(u, v, 0, 0)

    def get_normals(self, u, v):
        a, b = self.get_derivative(u, v, 1, 0), self.get_derivative(u, v, 0, 1)
        cross = np.cross(a.T, b.T).T
        return cross/np.linalg.norm(cross, axis=0)

    def get_curvature(self, u, v):
        a, b = self.get_derivative(u, v, 1, 0), self.get_derivative(u, v, 0, 1)
        n = self.get_normals(u, v)
        dot = lambda x, y: np.sum(x*y, axis=0)
        E, F, G = dot(a, a), dot(a, b), dot(b, b)
        L, M, N = [dot(n, self.get_derivative(u, v, *order)) for order in [(2, 0), (1, 1), (0, 2)]]
        return (E*N-2*F*M+G*L)/(2*(E*G-F*F)), (L*N-M*M)/(E*G-F*F)

    def project(self, points):
        results = []
        for point in points:
            def fun(uv):
                return float(np.sum((self.get_value([uv[0]], [uv[1]])[:, 0]-point)**2))
            candidates = []
            for u in np.linspace(0, 1, 5):
                for v in np.linspace(0, 1, 5):
                    r = minimize(fun, [u, v], method="L-BFGS-B", bounds=[(0, 1)]*2, options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 300})
                    candidates.extend((r.x, np.array([u, v])))
            for axis in (0, 1):
                for edge in (0., 1.):
                    uv = lambda t: np.array([edge, t]) if axis == 0 else np.array([t, edge])
                    r = minimize_scalar(lambda t: fun(uv(t)), bounds=(0, 1), method="bounded", options={"xatol": 1e-13})
                    candidates.append(uv(r.x))
            results.append(min(candidates, key=fun))
        return np.asarray(results)
