"""Independent checks for the scalable report experiments."""

import importlib.util
from pathlib import Path
import numpy as np
import nurbspy as nrb


def experiment():
    path = Path(__file__).parents[2] / "development/surface_fitting/_experiment.py"
    spec = importlib.util.spec_from_file_location("fitting_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sparse_evaluation_agrees_with_native_rational_surface():
    e = experiment()
    surface = e.flat(8)
    rng = np.random.default_rng(84)
    surface.P[2] = rng.normal(size=(8, 8))
    surface.W[:] = rng.uniform(0.5, 2.0, (8, 8))
    axis = np.linspace(0.0, 1.0, 17)
    u, v = np.meshgrid(axis, axis, indexing="ij")
    np.testing.assert_allclose(
        e.evaluate(surface, axis), surface.get_value(u.ravel(), v.ravel()).T, atol=2e-14
    )


def test_analytic_observations_are_reproducible_and_cover_boundaries():
    e = experiment()
    a, b = e.data(81, sigma=0.001), e.data(81, sigma=0.001)
    np.testing.assert_array_equal(a["points"], b["points"])
    assert a["axis"][0] == 0 and a["axis"][-1] == 1
    assert a["clean"][:, 2].min() < -0.3 and a["clean"][:, 2].max() > 0.4


def test_flat_initial_guesses_and_direct_refinement():
    e = experiment()
    observations = e.data(41)
    errors = []
    for count in (4, 8, 16, 32):
        initial = e.flat(count)
        assert np.all(initial.P[2] == 0) and np.all(initial.W == 1)
        fitted, info = e.linear(initial, observations)
        assert info["variables"] == 3 * count**2 and info["iterations"] == 0
        residual = e.evaluate(fitted, observations["axis"]) - observations["points"]
        np.testing.assert_allclose(
            e.basis(initial, observations["axis"]).T @ residual, 0, atol=1e-11
        )
        errors.append(e.metrics(fitted, observations)["clean_rms"])
        assert np.all(initial.P[2] == 0)
    assert np.all(np.diff(errors) < 0)


def test_standalone_html_embeds_assets_and_escapes_titles():
    figure = nrb.Figure(
        nrb.NurbsCurve([[0, 1], [0, 1], [0, 0]]),
        resolution=8,
        title="</title><script>bad()</script>",
    )
    html = figure.to_html(variants={"initial": figure, "final": figure.copy()})
    assert "&lt;/title&gt;" in html and "window.nurbspySelectVariant" in html
    assert "DecompressionStream" in html and "https://" not in html
