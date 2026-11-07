"""Regression checks for public Matplotlib helpers on current Matplotlib."""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from nurbspy import NurbsCurve, NurbsSurface
from tests.cases import paraboloid_control_points


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def geometry(kind, dimension):
    if kind == "curve":
        return NurbsCurve(np.array([[0., 1., 2.], [0., 1., 0.], [0., 0.2, 1.]])[:dimension])
    return NurbsSurface(paraboloid_control_points()[:dimension])


@pytest.mark.parametrize("kind", ["curve", "surface"])
@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_plot_uses_supplied_axes_or_figure(kind, dimension):
    obj = geometry(kind, dimension)
    is_3d = dimension == 3 or (kind == "surface" and dimension == 1)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d" if is_3d else None)
    assert obj.plot(ax=ax) == (fig, ax)
    assert len(fig.axes) == 1
    other = plt.figure()
    plotted, created = obj.plot(fig=other)
    assert plotted is other and created.figure is other
    assert created.name == ("3d" if is_3d else "rectilinear")
    fig.canvas.draw()


@pytest.mark.parametrize("kind", ["curve", "surface"])
def test_plot_rejects_mismatched_handles_and_projection(kind):
    obj = geometry(kind, 3)
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="3d axes"):
        obj.plot(ax=ax)
    other = plt.figure()
    ax3d = other.add_subplot(projection="3d")
    with pytest.raises(ValueError, match="same figure"):
        obj.plot(fig=fig, ax=ax3d)


@pytest.mark.parametrize("kind", ["curve", "surface"])
def test_supplied_axes_honour_visibility(kind):
    obj = geometry(kind, 3)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    obj.plot(ax=ax, ticks_off=True, axis_off=True)
    assert not ax.axison if ax.name != "3d" else not ax._axis3don
    assert not len(ax.get_xticks()) and not len(ax.get_yticks()) and not len(ax.get_zticks())


@pytest.mark.parametrize("method", ["plot_curvature", "plot_torsion"])
def test_curve_diagnostics_work_with_modern_ticks_and_supplied_axes(method):
    curve = NurbsCurve(np.array([[0., 1., 2., 3.], [0., 1., -1., 0.], [0., 0., 1., 1.]]))
    fig, ax = plt.subplots()
    assert getattr(curve, method)(ax=ax) == (fig, ax)
    fig.canvas.draw()
    assert np.all(np.isfinite(ax.lines[0].get_ydata()))


@pytest.mark.parametrize("mode", ["mean_curvature", "gaussian_curvature"])
def test_rectangular_curvature_mesh_and_colorbar_use_same_mapping(mode, monkeypatch):
    surface = NurbsSurface(paraboloid_control_points())
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    calls = []
    original = ax.plot_surface
    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)
    monkeypatch.setattr(ax, "plot_surface", capture)
    surface.plot(fig, ax, surface_color=mode, colorbar=True, Nu=7, Nv=13)
    args, kwargs = calls[0]
    assert args[0].shape == (7, 13)
    assert kwargs["rcount"] == 7 and kwargs["ccount"] == 13
    mappable = fig.axes[-1]._colorbar.mappable
    assert mappable.cmap.name == "viridis"
    np.testing.assert_allclose(kwargs["facecolors"], mappable.to_rgba(mappable.get_array()))
    np.testing.assert_allclose(fig.get_size_inches(), [8, 6])
    fig.canvas.draw()


def test_surface_normals_scale_with_geometry():
    lengths = []
    for scale in (1, 10):
        surface = NurbsSurface(paraboloid_control_points() * scale)
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        surface.plot_normals(fig, ax, number_u=1, number_v=1, scale=.1)
        segment = ax.collections[0]._segments3d[0]
        lengths.append(np.linalg.norm(segment[1] - segment[0]))
    np.testing.assert_allclose(lengths[1] / lengths[0], 10)


def test_degenerate_surface_normal_plot_skips_undefined_normals():
    surface = NurbsSurface(np.zeros((3, 2, 2)))
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        surface.plot_normals(fig, ax, number_u=2, number_v=2)
    assert len(ax.collections) == 0


@pytest.mark.parametrize("samples", [0, 1, 3.5, True])
def test_surface_sampling_errors_are_clear(samples):
    surface = geometry("surface", 3)
    with pytest.raises(ValueError, match="Nu must be an integer of at least 2"):
        surface.plot(Nu=samples)


def test_planar_curve_torsion_plot_is_finite():
    curve = NurbsCurve(np.array([[0., 1., 2., 3.], [0., 1., -1., 0.], [0., 0., 0., 0.]]))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        fig, ax = curve.plot_torsion()
        fig.canvas.draw()
    np.testing.assert_allclose(ax.lines[0].get_ydata(), 0)


def test_planar_frenet_frame_scale_changes_arrow_length():
    curve = NurbsCurve(np.array([[0., 1., 2.], [0., 1., 0.]]))
    vectors = []
    for scale in (.1, .2):
        fig, ax = plt.subplots()
        curve.plot_frenet_serret(fig, ax, frame_number=3, frame_scale=scale)
        arrow = ax.collections[0]
        vectors.append(np.hypot(arrow.U, arrow.V))
    np.testing.assert_allclose(vectors[1], 2 * vectors[0])


def test_curve_plot_rescale_false_preserves_supplied_limits():
    curve = geometry("curve", 3)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set(xlim=(-4, 4), ylim=(-3, 3), zlim=(-2, 2))
    curve.plot(ax=ax, rescale=False)
    assert ax.get_xlim() == (-4, 4) and ax.get_ylim() == (-3, 3) and ax.get_zlim() == (-2, 2)


def test_constant_curvature_keeps_surface_and_colorbar_consistent(monkeypatch):
    points = paraboloid_control_points()
    points[2] = 0
    surface = NurbsSurface(points)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    colors = []
    original = ax.plot_surface
    def capture(*args, **kwargs):
        colors.append(kwargs["facecolors"].copy())
        return original(*args, **kwargs)
    monkeypatch.setattr(ax, "plot_surface", capture)
    surface.plot(fig, ax, surface_color="mean_curvature", colorbar=True, Nu=4, Nv=5)
    mapper = fig.axes[-1]._colorbar.mappable
    np.testing.assert_allclose(colors[0], mapper.to_rgba(mapper.get_array()))


@pytest.mark.parametrize("dimension", [2, 3])
def test_straight_curve_frames_omit_undefined_normals(dimension):
    points = np.zeros((dimension, 3))
    points[0] = [0, 1, 2]
    curve = NurbsCurve(points)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d" if dimension == 3 else None)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        curve.plot_frenet_serret(fig, ax, frame_number=3)
        fig.canvas.draw()
    assert len(ax.collections) == 3  # Tangents exist; normals and binormals do not.
