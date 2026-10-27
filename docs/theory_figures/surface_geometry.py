"""Color an exact NURBS torus by actual mean/Gaussian curvature."""

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

from .common import torus, surface_grid, axis3d, save


def generate(output_dir):
    surface = torus()
    u, v, xyz = surface_grid(surface, 101)
    mean, gaussian = surface.get_curvature(u.ravel(), v.ravel())
    mean, gaussian = mean.reshape(u.shape), gaussian.reshape(u.shape)
    normals = surface.get_normals(u.ravel(), v.ravel()).reshape(xyz.shape)
    radius = np.hypot(xyz[0], xyz[1])
    cosine = (radius - 2.) / 0.6
    expected_k = cosine / (0.6 * (2 + 0.6*cosine))
    expected_h = -(2 + 1.2*cosine) / (1.2 * (2 + 0.6*cosine))
    np.testing.assert_allclose(gaussian, expected_k, atol=3e-13)
    np.testing.assert_allclose(mean, expected_h, atol=3e-13)
    fig = plt.figure(figsize=(11., 4.6), layout="constrained")
    for index, (field, cmap, title, label) in enumerate((
        (gaussian, "coolwarm", "(a) Gaussian curvature changes sign", "$K$ [length$^{-2}$]"),
        (mean, "viridis", "(b) Mean curvature uses the oriented normal", "$H$ [length$^{-1}$]"),
    ), start=1):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        norm = (colors.TwoSlopeNorm(vcenter=0., vmin=float(field.min()), vmax=float(field.max()))
                if index == 1 else colors.Normalize(float(field.min()), float(field.max())))
        ax.plot_surface(*xyz, facecolors=plt.colormaps[cmap](norm(field)),
                        shade=False, linewidth=0, rcount=101, ccount=101)
        if index == 2:
            # Actual outward normals, placed on the visible upper part of the torus.
            q = (slice(5, 95, 18), slice(6, 43, 12))
            ax.quiver(*(a[q] for a in xyz), *(a[q] for a in normals),
                       length=0.32, color="0.15", linewidth=0.8, normalize=True)
        axis3d(ax, xyz, elev=34, azim=-55)
        ax.set_title(title)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                     shrink=0.62, pad=0.10, label=label)
    save(fig, output_dir, "surface_geometry")
    return {"mean_curvature_reference_error": float(np.max(abs(mean - expected_h))),
            "gaussian_curvature_reference_error": float(np.max(abs(gaussian - expected_k))),
            "normal_length_error": float(np.max(abs(np.linalg.norm(normals, axis=0) - 1)))}
