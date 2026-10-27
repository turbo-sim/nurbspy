"""Integrate derivative-based Jacobians; no nonexistent surface-area API."""

import matplotlib.pyplot as plt
from matplotlib import colors
from numpy.polynomial.legendre import leggauss
import numpy as np
from scipy.integrate import dblquad

from .common import BLUE, ORANGE, paraboloid, surface_grid, axis3d, save


def integrate(surface, order):
    """Tensor Gauss rule over this single-span patch, assembled from public APIs."""
    nodes, weights = leggauss(order)
    u, v = np.meshgrid((nodes + 1) / 2, (nodes + 1) / 2, indexing="ij")
    w = np.outer(weights, weights).ravel() / 4
    xyz = surface.get_value(u.ravel(), v.ravel())
    su = surface.get_derivative(u.ravel(), v.ravel(), 1, 0)
    sv = surface.get_derivative(u.ravel(), v.ravel(), 0, 1)
    vector_area = np.cross(su, sv, axis=0)
    jacobian = np.linalg.norm(vector_area, axis=0)
    density = 1 + (xyz[0]**2 + xyz[1]**2) / 2
    return float(w @ jacobian), float(w @ (density * jacobian)), float(w @ vector_area[2])


def generate(output_dir):
    surface = paraboloid()
    orders = np.array([2, 4, 8, 16, 32])
    values = np.array([integrate(surface, int(n)) for n in orders])
    # Independent graph formulas in physical x,y coordinates.
    reference_area = dblquad(lambda y, x: np.sqrt(1+x*x+y*y), -1., 1., -1., 1.,
                             epsabs=1e-11, epsrel=1e-11)[0]
    reference_mass = dblquad(lambda y, x: (1+(x*x+y*y)/2)*np.sqrt(1+x*x+y*y),
                             -1., 1., -1., 1., epsabs=1e-11, epsrel=1e-11)[0]
    _, _, xyz = surface_grid(surface)
    density = 1 + (xyz[0]**2 + xyz[1]**2) / 2
    norm = colors.Normalize(1, 2)
    fig = plt.figure(figsize=(10.6, 4.3), layout="constrained")
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(*xyz, facecolors=plt.colormaps["viridis"](norm(density)),
                    shade=False, linewidth=0, rcount=81, ccount=81)
    axis3d(ax, xyz, elev=28, azim=-52)
    ax.set_title("(a) A scalar field on an exact quadratic patch")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax,
                 shrink=0.63, pad=0.10, label="Surface density $\\rho$")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.semilogy(orders, np.maximum(abs(values[:, 0] - reference_area), 1e-15),
                 "o-", color=BLUE, label="Area")
    ax2.semilogy(orders, np.maximum(abs(values[:, 1] - reference_mass), 1e-15),
                 "s--", color=ORANGE, label="Density integral")
    ax2.set(xlabel="Gauss nodes per parameter axis", ylabel="Absolute integral error",
            title="(b) Check convergence of the integral")
    ax2.legend()
    save(fig, output_dir, "surface_integrals")
    assert abs(values[-1, 0] - reference_area) < 1e-10
    assert abs(values[-1, 1] - reference_mass) < 1e-10
    np.testing.assert_allclose(values[:, 2], 4., atol=2e-13)
    return {"area": reference_area, "density_integral": reference_mass,
            "upward_flux": float(values[-1, 2]), "orders": orders.tolist(),
            "computed_integrals": values.tolist()}
