"""The optional JAX backend shares host rendering with the NumPy backend."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

jax = pytest.importorskip("nurbspy.jax")
from tests.cases import rational_curve_data, paraboloid_control_points


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_jax_curve_uses_supplied_axes_and_retains_rescale_option():
    curve = jax.NurbsCurve(**rational_curve_data())
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set(xlim=(-4, 4), ylim=(-3, 3), zlim=(-2, 2))
    assert curve.plot(ax=ax, rescale=False) == (fig, ax)
    assert ax.get_xlim() == (-4, 4) and ax.get_ylim() == (-3, 3) and ax.get_zlim() == (-2, 2)
    assert len(fig.axes) == 1
    fig.canvas.draw()


def test_jax_curve_styling_and_parameter_subinterval_are_preserved():
    curve = jax.NurbsCurve(**rational_curve_data())
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    assert curve.plot_curve(fig, ax, linewidth=3, linestyle="--", color="purple", u1=.2, u2=.7) == (fig, ax)
    line = ax.lines[0]
    assert line.get_color() == "purple" and line.get_linestyle() == "--" and line.get_linewidth() == 3
    np.testing.assert_allclose(np.array(line.get_data_3d())[:, 0], np.asarray(curve.get_value(.2))[:, 0], atol=1e-12)
    np.testing.assert_allclose(np.array(line.get_data_3d())[:, -1], np.asarray(curve.get_value(.7))[:, 0], atol=1e-12)


@pytest.mark.parametrize("method", ["plot_curvature", "plot_torsion"])
def test_jax_curve_diagnostics_inherit_current_matplotlib_support(method):
    curve = jax.NurbsCurve(**rational_curve_data())
    fig, ax = plt.subplots()
    assert getattr(curve, method)(ax=ax, color="purple", linestyle="--") == (fig, ax)
    assert ax.lines[0].get_color() == "purple"
    fig.canvas.draw()


@pytest.mark.parametrize("kind", ["curve", "surface"])
def test_plotting_does_not_mutate_or_rebind_jax_geometry(kind):
    geometry = (jax.NurbsCurve(**rational_curve_data()) if kind == "curve"
                else jax.NurbsSurface(paraboloid_control_points()))
    points, weights = geometry.P, geometry.W
    before = np.asarray(points).copy()
    fig, ax = geometry.plot()
    fig.canvas.draw()
    assert geometry.P is points and geometry.W is weights
    np.testing.assert_array_equal(geometry.P, before)
