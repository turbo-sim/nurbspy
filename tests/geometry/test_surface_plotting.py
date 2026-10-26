"""Host-side plotting remains usable for either numerical backend."""

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_surface_plotting_dimensions(surface_backend, dimension):
    """Draw scalar, planar and spatial patches with their public plot options."""
    points = np.array(
        [[[0.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]], [[0.0, 0.2], [0.1, 0.4]]]
    )
    surface = surface_backend.NurbsSurface(points[:dimension])
    fig = None
    try:
        fig, ax = surface.plot(
            control_points=True,
            Nu=5,
            Nv=6,
            isocurves_u=3 if dimension > 1 else None,
            isocurves_v=3 if dimension > 1 else None,
        )
        if dimension == 3:
            surface.plot_normals(fig, ax, number_u=2, number_v=2)
            surface.rescale_plot(fig, ax)
        fig.canvas.draw()
        assert ax.has_data()
    finally:
        plt.close("all")


@pytest.mark.parametrize("curvature", ["mean", "gaussian"])
def test_curvature_plot_and_constant_colors(surface_backend, curvature):
    """Plot a planar patch's zero curvature without invalid color normalization."""
    points = np.array(
        [[[0.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]]
    )
    surface = surface_backend.NurbsSurface(points)
    try:
        fig, ax = surface.plot_curvature(curvature_type=curvature)
        fig.canvas.draw()
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            fig, ax = surface.plot(
                surface_color=curvature + "_curvature", colorbar=True, Nu=5, Nv=6
            )
            fig.canvas.draw()
        assert ax.has_data()
    finally:
        plt.close("all")
