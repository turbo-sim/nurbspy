"""Automatic sampling accuracy, bounded previews, and viewer transport contracts."""

from io import StringIO
import base64
import os

import numpy as np
import pytest
import nurbspy as nrb
from nurbspy.visualization.model import _GeometryModel
from nurbspy.visualization.sampling import automatic_axes, MAX_AXIS, MAX_VERTICES
from tests.geometry._editor_session import (
    editor_page,
    wait_ready,
    points,
    history_button,
)


def circle():
    return nrb.CircularArc(
        nrb.Point(0, 0, 0),
        np.array([1.0, 0, 0]),
        np.array([0.0, 1, 0]),
        1,
        0,
        2 * np.pi,
    ).NurbsCurve


def plane():
    x, y = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4), indexing="ij")
    return nrb.NurbsSurface(np.stack((x, y, x * 0)))


def packet(model, quality="full", **extra):
    return dict(
        revision=1,
        action="edit",
        quality=quality,
        control_points=model.initial_state["control_points"],
        weights=model.initial_state["weights"],
        **extra,
    )


def test_flat_and_directionally_curved_surfaces_use_different_densities():
    u, v, _ = automatic_axes(plane())
    assert len(u) * len(v) == 64
    cylinder = nrb.NurbsSurfaceExtruded(circle(), np.array([0.0, 0, 1]), 2).NurbsSurface
    u, v, _ = automatic_axes(cylinder)
    assert len(u) > 4 * len(v) and len(u) * len(v) < 40**2
    saddle = plane()
    saddle.P[2] = saddle.P[0] * saddle.P[1]
    u, v, _ = automatic_axes(saddle)
    assert len(u) > 8 and len(v) > 8


def test_circle_presets_are_nested_and_control_chord_error():
    geometry = circle()
    counts = []
    for quality, tolerance in [("fast", 0.003), ("auto", 0.0007), ("fine", 0.00018)]:
        u, _, _ = automatic_axes(geometry, quality)
        counts.append(len(u))
        actual = geometry.get_value((u[:-1] + u[1:]) / 2)
        chord = (geometry.get_value(u[:-1]) + geometry.get_value(u[1:])) / 2
        scale = np.linalg.norm(np.ptp(geometry.P, axis=1))
        assert np.max(np.linalg.norm(actual - chord, axis=0)) < tolerance * scale
    assert counts[0] < counts[1] < counts[2]


def test_knots_and_sampling_are_scale_invariant():
    original = circle()
    u, _, _ = automatic_axes(original)
    assert set(np.unique(original.U)) <= set(u)
    for scale in (1e-5, 1e5):
        geometry = circle()
        geometry.P *= scale
        scaled, _, _ = automatic_axes(geometry)
        np.testing.assert_array_equal(scaled, u)


def test_sampling_and_preview_budgets_are_bounded():
    geometry = plane()
    geometry.P[2, 1:3, 1:3] = 50
    model = _GeometryModel(geometry, id="budget", quality="fine")
    full = model._initial_scene["topologies"]["full"]
    assert full["count"] <= MAX_VERTICES
    assert all(count <= MAX_AXIS for count in full["shape"])
    assert model._initial_scene["topologies"]["preview"]["count"] <= 20**2


def test_drag_preview_reuses_client_sampling_and_full_edit_refines(monkeypatch):
    model = _GeometryModel(plane(), id="edit")
    request = packet(model, "preview", sampling=model._description)
    request["control_points"][2][1][1] = 2
    choose = model._choose_sampling
    monkeypatch.setattr(
        model, "_choose_sampling", lambda *args: pytest.fail("Preview refined the grid")
    )
    preview = model.render_packet(request)
    assert len(base64.b64decode(preview["vertices"])) == 64 * 3 * 4
    monkeypatch.setattr(model, "_choose_sampling", choose)
    request["quality"] = "full"
    result = model.render_packet(request)
    assert result["topologies"]["full"]["count"] > 64
    inspection = model.inspect(
        dict(
            revision=2,
            quality="full",
            field="u",
            sampling=result["sampling"],
            control_points=request["control_points"],
            weights=request["weights"],
        )
    )
    assert len(inspection["field"]["values"]) == result["topologies"]["full"]["count"]
    # A second browser's original grid remains valid after the first refines.
    untouched = model.render_packet(
        packet(model, "preview", sampling=model._description)
    )
    assert len(base64.b64decode(untouched["vertices"])) == 64 * 3 * 4


@pytest.mark.parametrize("geometry", [circle, plane])
@pytest.mark.parametrize("resolution", [None, 32])
def test_full_previews_keep_the_final_sampling_without_refining(geometry, resolution, monkeypatch):
    model = _GeometryModel(geometry(), id="full-preview", resolution=resolution,
                           quality="fine", preview_quality="full")
    scene = model._initial_scene
    assert scene["topologies"]["preview"] == scene["topologies"]["full"]
    if scene["sampling"] is not None:
        assert scene["sampling"]["preview"] == scene["sampling"]["full"]
    monkeypatch.setattr(model, "_choose_sampling", lambda *args: pytest.fail("Preview refined the grid"))
    preview = model.render_packet(packet(model, "preview", sampling=scene["sampling"]))
    assert len(base64.b64decode(preview["vertices"])) == scene["topologies"]["full"]["count"] * 12


@pytest.mark.parametrize("quality", ["fine", "bad", None, 5, True, []])
def test_invalid_preview_settings_are_rejected(quality):
    with pytest.raises(ValueError, match="preview_quality"):
        nrb.Figure(circle(), preview_quality=quality)


def test_full_preview_policy_round_trips_and_legacy_views_keep_coarse_previews():
    figure = nrb.Figure(circle(), quality="fine", preview_quality="full")
    restored = nrb.Figure.from_yaml(StringIO(figure.to_yaml()))
    assert restored.preview_quality == "full"
    assert restored.to_state() == figure.to_state()
    state = figure.to_state()
    state.pop("preview_quality")
    legacy = nrb.Figure.from_state(state)
    assert legacy.preview_quality == "coarse"
    scene = legacy._models()[legacy.active]._initial_scene
    assert scene["topologies"]["preview"]["count"] < scene["topologies"]["full"]["count"]
    with pytest.raises(ValueError, match="preview_resolution"):
        nrb.Figure(circle(), preview_quality="full", preview_resolution=20)


@pytest.mark.parametrize("read_only", [False, True])
def test_embedded_figure_propagates_full_preview_policy(read_only):
    from nurbspy.visualization import GeometryView, GeometryEditor

    figure = nrb.Figure(circle(), quality="fine")
    component = (GeometryView if read_only else GeometryEditor)(
        figure, id="full-preview", preview_quality="full")
    assert component.preview_quality == component._figure.preview_quality == "full"
    assert figure.preview_quality == "coarse"
    scene = component._initial_scene
    assert scene["topologies"]["preview"] == scene["topologies"]["full"]


@pytest.mark.parametrize("method", ["show", "edit"])
@pytest.mark.parametrize("use_figure", [False, True])
def test_viewing_and_editing_functions_accept_preview_policy(method, use_figure, monkeypatch):
    from nurbspy.visualization import figure_hosts

    calls = []
    def open_figure(figure, **options):
        calls.append((figure.preview_quality, figure.quality, options["read_only"]))
        return None
    monkeypatch.setattr(figure_hosts, "open_figure", open_figure)
    geometry = nrb.Figure(circle(), quality="fine") if use_figure else circle()
    options = {"preview_quality": "full"}
    if not use_figure:
        options["quality"] = "fine"
    getattr(geometry, method)(**options)
    assert calls == [("full", "fine", method == "show")]
    if use_figure:
        # Cancelling an edit preserves the original figure's preview setting.
        assert geometry.preview_quality == ("full" if method == "show" else "coarse")


@pytest.mark.parametrize("quality", ["bad", None, 5, True])
def test_invalid_presets_are_rejected(quality):
    with pytest.raises(ValueError, match="quality"):
        nrb.Figure(plane(), quality=quality)


def test_old_numeric_sampling_and_new_quality_state_round_trip():
    model = _GeometryModel(plane(), id="legacy", resolution=12)
    assert model._initial_scene["topologies"]["full"]["count"] == 144
    figure = nrb.Figure(circle(), quality="fine")
    assert figure.resolution is None and figure.settings()["quality"] == "fine"
    restored = nrb.Figure.from_yaml(StringIO(figure.to_yaml()))
    assert restored.to_state() == figure.to_state()
    legacy = nrb.Figure(circle(), resolution=12).to_state()
    legacy.pop("quality")
    assert nrb.Figure.from_state(legacy).resolution == 12


def test_concurrent_refinement_keeps_grids_consistent_and_cache_bounded():
    from concurrent.futures import ThreadPoolExecutor

    model = _GeometryModel(plane(), id="shared")

    def refine(height):
        request = packet(model)
        request["control_points"][2][1][1] = height
        result = model.render_packet(request)
        assert (
            len(base64.b64decode(result["vertices"]))
            == result["topologies"]["full"]["count"] * 12
        )
        return result["sampling"]

    with ThreadPoolExecutor(max_workers=4) as executor:
        assert len(list(executor.map(refine, np.linspace(0.1, 2, 12)))) == 12
    assert len(model._sample_cache) <= 6
    assert np.count_nonzero(model.initial_state["control_points"][2]) == 0


@pytest.mark.skipif(
    os.environ.get("NURBSPY_TEST_PNG") != "1", reason="Explicit browser check"
)
@pytest.mark.parametrize(
    "editor_page", ["desktop_auto", "embedded_auto"], indirect=True
)
def test_automatic_quality_controls_and_edits(editor_page):
    page = editor_page
    page.get_by_role("button", name="Open display settings", exact=True).click()
    selector = page.get_by_role("combobox", name="Display quality", exact=True)
    assert selector.input_value() == "auto" and selector.is_enabled()
    camera = page.evaluate("testState().camera")
    selector.select_option("fine")
    wait_ready(page)
    fine = int(page.locator(".nrb-web").first.get_attribute("data-sample-count"))
    selector.select_option("fast")
    wait_ready(page)
    fast = int(page.locator(".nrb-web").first.get_attribute("data-sample-count"))
    assert fast <= fine
    assert page.evaluate("testState().camera") == camera
    assert page.evaluate("testState().objects.surface.display.quality") == "fast"
    assert page.evaluate("testState().quality") == "fast"
    before = points(page)
    page.get_by_role("button", name="Open editing controls", exact=True).click()
    coordinate = page.get_by_role("spinbutton", name="Z coordinate", exact=True)
    coordinate.fill("1.5")
    coordinate.press("Tab")
    wait_ready(page)
    assert not np.array_equal(points(page), before)
    history_button(page).click()
    wait_ready(page)
    np.testing.assert_array_equal(points(page), before)
    x, y = page.evaluate("testPositions()[4]")
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + 20, y - 10, steps=4)
    page.mouse.up()
    wait_ready(page)
    assert np.linalg.norm(points(page) - before) > 0.001


@pytest.mark.skipif(
    os.environ.get("NURBSPY_TEST_PNG") != "1", reason="Explicit browser check"
)
def test_offline_html_carries_working_quality_presets(tmp_path):
    from playwright.sync_api import sync_playwright
    from nurbspy.visualization.figure_hosts import _launch_browser

    figure = nrb.Figure(plane(), name="surface", display=dict(quantity="u"))
    figure.add(circle(), name="curve", display=dict(quantity="u"))
    document = tmp_path / "offline.html"
    figure.to_html(document)
    with sync_playwright() as playwright:
        browser = _launch_browser(playwright)
        page = browser.new_page()
        errors, network = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "request",
            lambda request: (
                network.append(request.url)
                if request.url.startswith(("http:", "https:"))
                else None
            ),
        )
        page.goto(document.as_uri())
        page.wait_for_function("window.nurbspyReportReady === true")
        page.evaluate(
            "async()=>{await window.nurbspyReady;await window.nurbspyFigure.ready();}"
        )
        for name in ("surface", "curve"):
            if name == "curve":
                page.get_by_role(
                    "combobox", name="Figure object", exact=True
                ).select_option(name)
                page.evaluate("async()=>await window.nurbspyFigure.ready()")
            page.get_by_role("button", name="Open display settings", exact=True).click()
            for preset in ("fine", "fast", "auto"):
                page.get_by_role(
                    "combobox", name="Display quality", exact=True
                ).select_option(preset)
                page.wait_for_function(
                    "()=>!document.querySelector('.nrb-web').classList.contains('nrb-updating')"
                )
                page.evaluate("async()=>await window.nurbspyFigure.ready()")
                state = page.evaluate("window.nurbspyFigure.snapshot()")
                assert state["objects"][name]["display"]["quality"] == preset
            assert page.locator(".nrb-colorbar").first.is_visible()
        assert not errors and not network
        browser.close()


@pytest.mark.skipif(
    os.environ.get("NURBSPY_TEST_PNG") != "1", reason="Explicit browser check"
)
@pytest.mark.parametrize("read_only", [True, False])
def test_direct_embedded_components_support_automatic_sampling(read_only):
    import threading
    from dash import Dash
    from werkzeug.serving import make_server
    from playwright.sync_api import sync_playwright
    from nurbspy.visualization import GeometryView, GeometryEditor
    from nurbspy.visualization.figure_hosts import _launch_browser

    cylinder = nrb.NurbsSurfaceExtruded(circle(), np.array([0.0, 0, 1]), 2).NurbsSurface
    app = Dash(__name__)
    component = (GeometryView if read_only else GeometryEditor)(
        cylinder, id="direct-auto"
    )
    app.layout = component.layout
    component.register(app)
    server = make_server("127.0.0.1", 0, app.server, threaded=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as playwright:
            browser = _launch_browser(playwright)
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}")
            page.locator("canvas.nrb-renderer").wait_for()
            page.get_by_role("button", name="Open display settings", exact=True).click()
            selector = page.get_by_role("combobox", name="Display quality", exact=True)
            counts = []
            for preset in ("fine", "fast"):
                selector.select_option(preset)
                page.wait_for_function(
                    "()=>!document.querySelector('.nrb-web').classList.contains('nrb-updating')"
                )
                counts.append(
                    int(page.locator(".nrb-web").get_attribute("data-sample-count"))
                )
            assert counts[0] > counts[1]
            if not read_only:
                page.get_by_role(
                    "button", name="Open editing controls", exact=True
                ).click()
                coordinate = page.get_by_role(
                    "spinbutton", name="Z coordinate", exact=True
                )
                coordinate.fill("0.25")
                coordinate.press("Tab")
                page.wait_for_function(
                    "()=>!document.querySelector('.nrb-web').classList.contains('nrb-updating')"
                )
                assert not page.locator(".nrb-error").count()
            assert not errors
            browser.close()
    finally:
        server.shutdown()
