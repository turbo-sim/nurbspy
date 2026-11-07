"""Regression coverage for shared Matplotlib styling and export helpers."""
import builtins
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import nurbspy.graphics as graphics


@pytest.mark.parametrize("name", ["python", "default", "PYTHON", "matlab"])
def test_named_color_cycles(name):
    with matplotlib.rc_context():
        graphics.set_plot_options(color_order=name)
        expected = graphics.COLORS_MATLAB if name == "matlab" else graphics.COLORS_PYTHON
        assert matplotlib.rcParams["axes.prop_cycle"].by_key()["color"] == expected


def test_custom_color_cycle():
    with matplotlib.rc_context():
        colors = ["royalblue", "#ff8800", (0.2, 0.4, 0.6)]
        graphics.set_plot_options(color_order=iter(colors))
        assert matplotlib.rcParams["axes.prop_cycle"].by_key()["color"] == colors


@pytest.mark.parametrize("colors", ["unknown", [], ["not a color"], None, 7])
def test_invalid_color_cycle_fails_before_changing_style(colors):
    before = matplotlib.rcParams["axes.prop_cycle"]
    with pytest.raises(ValueError, match="color_order"):
        graphics.set_plot_options(color_order=colors)
    assert matplotlib.rcParams["axes.prop_cycle"] == before


def test_box_is_reused_and_updates_style():
    fig = plt.figure()
    try:
        ax = fig.add_subplot(projection="3d")
        first = graphics.close_3d_box(ax)
        second = graphics.close_3d_box(ax, color="red", linewidth=2, alpha=0.5)
        assert second is first
        assert len(ax.collections) == 1
        np.testing.assert_allclose(second.get_colors()[0], matplotlib.colors.to_rgba("red", 0.5))
        np.testing.assert_allclose(second.get_linewidths(), [2])
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_box_does_not_affect_autoscaling():
    fig = plt.figure()
    try:
        ax = fig.add_subplot(projection="3d")
        ax.plot([0, 1], [0, 1], [0, 1])
        before = (ax.xy_dataLim.extents.copy(), ax.zz_dataLim.extents.copy())
        ax.set(xlim=(-100, 100), ylim=(-100, 100), zlim=(-100, 100))
        graphics.close_3d_box(ax)
        np.testing.assert_array_equal(ax.xy_dataLim.extents, before[0])
        np.testing.assert_array_equal(ax.zz_dataLim.extents, before[1])
        fig.canvas.draw()
        np.testing.assert_array_equal(ax.xy_dataLim.extents, before[0])
        np.testing.assert_array_equal(ax.zz_dataLim.extents, before[1])
    finally:
        plt.close(fig)


def test_box_can_be_recreated_after_removal_or_clear():
    fig = plt.figure()
    try:
        ax = fig.add_subplot(projection="3d")
        first = graphics.close_3d_box(ax)
        first.remove()
        second = graphics.close_3d_box(ax)
        assert second is not first
        ax.clear()
        third = graphics.close_3d_box(ax)
        assert third is not second
        assert len(ax.collections) == 1
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_optional_export_missing_imageio_is_actionable(monkeypatch):
    monkeypatch.setattr(graphics, "IMAGEIO_AVAILABLE", False)
    for export, output in [(graphics.create_gif, "movie.gif"), (graphics.create_mp4, "movie.mp4")]:
        with pytest.raises(ImportError, match="pip install imageio"):
            export("missing-images", output)


def test_graphics_imports_without_imageio(monkeypatch):
    original_import = builtins.__import__

    def without_imageio(name, *args, **kwargs):
        if name == "imageio" or name.startswith("imageio."):
            raise ModuleNotFoundError("imageio unavailable", name="imageio")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_imageio)
    spec = importlib.util.spec_from_file_location("graphics_without_imageio", Path(graphics.__file__))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.IMAGEIO_AVAILABLE is False
    with pytest.raises(ImportError, match="pip install imageio"):
        module.create_gif("missing-images", "movie.gif")
