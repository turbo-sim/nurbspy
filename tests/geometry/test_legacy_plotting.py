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














def test_curve_plot_rescale_false_preserves_supplied_limits():
    curve = geometry("curve", 3)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set(xlim=(-4, 4), ylim=(-3, 3), zlim=(-2, 2))
    curve.plot(ax=ax, rescale=False)
    assert ax.get_xlim() == (-4, 4) and ax.get_ylim() == (-3, 3) and ax.get_zlim() == (-2, 2)


