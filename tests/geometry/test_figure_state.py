"""Reproducible figure state and transactional host behavior."""

from io import BytesIO, StringIO
import os

import numpy as np
import pytest

import nurbspy as nrb
from nurbspy.visualization.figure_hosts import _FigureBridge


def curve():
    return nrb.NurbsCurve([nrb.Point(-1, 0, 0), nrb.Point(0, 2, .5), nrb.Point(1, 0, 0)])


def surface():
    return nrb.NurbsSurface([[nrb.Point(-1, -1, 0), nrb.Point(-1, 1, .5)],
                             [nrb.Point(1, -1, 0), nrb.Point(1, 1, 0)]])


def test_multiple_objects_round_trip_geometry_camera_and_all_display_settings():
    figure = nrb.Figure(curve(), name="curve", resolution=12, background="#fafafa")
    figure.add(surface(), name="surface", display=dict(quantity="mean_curvature",
               colormap="coolwarm", reverse=True, limits=[-2., 2.], control_points=False,
               control_polygon=False, boundary=False, mesh="display"), visible=False)
    figure.set_camera(position=[4, -5, 6], target=[0, 0, 0], up=[0, 0, 1], zoom=1.2)
    restored = nrb.Figure.from_yaml(StringIO(figure.to_yaml()))
    assert restored.to_state() == figure.to_state()
    assert restored.names == ("curve", "surface")
    restored.update("curve", visible=False)
    assert figure.to_state()["objects"]["curve"]["visible"]


def test_objects_are_copies_and_scripted_updates_are_explicit():
    original = curve()
    figure = nrb.Figure(original, resolution=8)
    points = original.P.copy()
    original.P[0, 0] = 20
    np.testing.assert_array_equal(figure.geometry().P, points)
    changed = figure.geometry()
    changed.P[2, 1] = 2
    np.testing.assert_array_equal(figure.geometry().P, points)
    figure.update(geometry=changed, display={"grid": False})
    assert figure.geometry().P[2, 1] == 2
    assert not figure.settings()["grid"]
    assert not hasattr(original, "camera")


@pytest.mark.parametrize("projection,axis,up", [("xy", 2, [0, 1, 0]), ("yz", 0, [0, 0, 1]), ("zx", 1, [1, 0, 0])])
def test_scripted_planar_camera_defaults_look_along_the_named_axis(projection, axis, up):
    figure = nrb.Figure(curve(), resolution=8).set_camera(projection=projection)
    direction = np.subtract(figure.camera["position"], figure.camera["target"])
    assert direction[axis] > 0
    assert np.count_nonzero(direction) == 1
    assert figure.camera["up"] == up


@pytest.mark.parametrize("mutate", [
    lambda s: s.update(version=99),
    lambda s: s.update(active="missing"),
    lambda s: s["rendering"].update(width=True),
    lambda s: s["rendering"].update(background="invalid"),
    lambda s: s.update(camera={"position": [0, 0, 0], "target": [0, 0, 0]}),
    lambda s: s["objects"]["geometry"]["display"].update(limits=[2, 1]),
    lambda s: s["objects"]["geometry"]["display"].update(reverse="yes"),
])
def test_malformed_figure_states_are_rejected(mutate):
    state = nrb.Figure(curve(), resolution=8).to_state()
    mutate(state)
    with pytest.raises((ValueError, TypeError)):
        nrb.Figure.from_state(state)


def test_geometry_yaml_and_figure_yaml_are_distinct_formats():
    with pytest.raises(ValueError):
        nrb.Figure.from_yaml(StringIO(curve().to_yaml()))
    with pytest.raises(ValueError):
        nrb.NurbsCurve.from_yaml(StringIO(nrb.Figure(curve()).to_yaml()))


def test_empty_figures_and_duplicate_names_fail_without_losing_objects():
    figure = nrb.Figure()
    with pytest.raises(ValueError):
        figure.to_state()
    figure.add(curve(), name="line")
    with pytest.raises(ValueError):
        figure.add(curve(), name="line")
    assert figure.names == ("line",)


def test_edit_apply_commits_and_cancel_leaves_figure_unchanged():
    figure = nrb.Figure(curve(), resolution=8)
    before = figure.to_state()
    bridge = _FigureBridge(figure, False)
    bridge._window = type("Window", (), {"destroy": lambda self: None})()
    changed = figure.copy().update(display={"geometry_color": "#ff0000"}).to_state()
    bridge.capture(changed)
    assert figure.to_state() == before
    assert bridge.finish(None, False) == {"ok": True}
    assert figure.to_state() == before
    bridge = _FigureBridge(figure, False)
    bridge._window = type("Window", (), {"destroy": lambda self: None})()
    assert bridge.finish(changed, True) == {"ok": True}
    assert figure.settings()["geometry_color"] == "#ff0000"


def test_viewer_rejects_geometry_changes_and_editor_rejects_topology_changes():
    figure = nrb.Figure(curve(), resolution=8)
    changed = figure.geometry()
    changed.P[2, 1] = 10
    edited = figure.copy().update(geometry=changed).to_state()
    viewer = _FigureBridge(figure, True)
    viewer._window = type("Window", (), {"destroy": lambda self: None})()
    assert "error" in viewer.finish(edited, True)
    assert not viewer._closed
    editor = _FigureBridge(figure, False)
    editor._window = viewer._window
    changed_topology = figure.copy().update(geometry=nrb.NurbsCurve([
        nrb.Point(0, 0, 0), nrb.Point(1, 1, 0)]))
    assert "error" in editor.finish(changed_topology.to_state(), True)


def test_native_bridge_exposes_no_window_or_geometry_attributes():
    bridge = _FigureBridge(nrb.Figure(curve(), resolution=8), True)
    assert all(name.startswith("_") for name in vars(bridge))


def test_native_bootstrap_keeps_scene_data_out_of_navigate_to_string():
    import base64
    import gzip
    import json
    from nurbspy.visualization.figure_hosts import figure_html

    figure = nrb.Figure(surface(), resolution=70).add(curve(), name="reference")
    html = figure_html(figure, inline=False)
    assert len(html.encode("utf-16-le")) < 16000
    assert "const packed=null" in html and "api.startup_data()" in html
    bridge = _FigureBridge(figure, True)
    packed = bridge.startup_data()
    assert bridge.startup_data() is packed
    startup = json.loads(gzip.decompress(base64.b64decode(packed)))
    assert startup["payload"]["state"] == figure.to_state()
    assert "nurbspy/figure" in startup["modules"]["imports"]


def test_title_bar_close_does_not_evaluate_javascript_on_the_gui_thread():
    import threading

    bridge = _FigureBridge(nrb.Figure(curve(), resolution=8), True)
    called = threading.Event()
    caller = threading.get_ident()
    threads = []

    class Window:
        def evaluate_js(self, script):
            threads.append(threading.get_ident())
            called.set()

    bridge._window = Window()
    assert bridge.closing() is False
    assert called.wait(2)
    assert threads == [threads[0]] and threads[0] != caller


def test_dash_figure_stores_are_independent_and_routes_match_mode():
    from dash import Dash, html
    from nurbspy.visualization import GeometryView, GeometryEditor

    figure = nrb.Figure(curve(), name="curve", resolution=8).add(surface(), name="surface")
    app = Dash(__name__, requests_pathname_prefix="/research/", routes_pathname_prefix="/research/")
    view = GeometryView(figure, id="view")
    edit = GeometryEditor(figure, id="edit")
    app.layout = html.Div([view.layout, edit.layout])
    view.register(app)
    edit.register(app)
    client = app.server.test_client()
    assert client.get("/research/_dash-layout").status_code == 200
    assert client.get("/research/_nurbspy_web/figure.js").status_code == 200
    routes = {rule.rule for rule in app.server.url_map.iter_rules()}
    assert "/research/_nurbspy_web/figure/edit/edit" in routes
    assert "/research/_nurbspy_web/figure/view/edit" not in routes
    before = figure.to_state()
    state = figure.copy().update("curve", visible=False).to_state()
    restored = view.figure_from_state(state)
    assert not restored.to_state()["objects"]["curve"]["visible"]
    assert figure.to_state() == before
    request = edit._figure._models()["curve"].initial_state
    response = client.post("/research/_nurbspy_web/figure/edit/edit", json={"name": "curve", "packet": {
        "revision": 1, "quality": "full", "action": "edit", "control_points": request["control_points"],
        "weights": request["weights"]}})
    assert response.status_code == 200 and response.mimetype == "application/octet-stream"
    assert client.post("/research/_nurbspy_web/figure/view/scenes", json=state).status_code == 200


def test_unsupported_image_formats_and_bad_dimensions_fail_before_browser_launch():
    figure = nrb.Figure(curve(), resolution=8)
    for name in ("image.svg", "image.pdf", "image.jpeg"):
        with pytest.raises(ValueError, match="PNG"):
            figure.savefig(name)
    with pytest.raises(ValueError):
        figure.savefig(BytesIO(), width=0, height=200)


def test_frame_export_rejects_unsafe_prefixes_before_launch(tmp_path):
    figure = nrb.Figure(curve(), resolution=8)
    with pytest.raises(ValueError, match="prefix"):
        figure.saveframes([], tmp_path, prefix="../outside")
    with pytest.raises(TypeError, match="progress"):
        figure.saveframes([], tmp_path, progress=True)


@pytest.mark.skipif(os.environ.get("NURBSPY_TEST_PNG") != "1", reason="Explicit headless browser integration check")
def test_real_png_round_trip_and_geometry_animation(tmp_path):
    from PIL import Image, ImageChops

    figure = nrb.Figure(surface(), name="surface", resolution=12,
                        display={"control_points": False, "control_polygon": False})
    figure.add(curve(), name="curve", display=dict(geometry_color="#ff3300", control_polygon=False))
    first = tmp_path / "frame000.png"
    second = tmp_path / "frame001.png"
    figure.savefig(first, width=800, height=600)
    assert Image.open(first).size == (800, 600)
    assert Image.open(first).convert("RGB").getextrema() != ((255, 255),) * 3
    restored = nrb.Figure.from_yaml(StringIO(figure.to_yaml()))
    restored.savefig(second, width=800, height=600)
    difference = np.asarray(ImageChops.difference(Image.open(first).convert("RGB"), Image.open(second).convert("RGB")))
    assert difference.mean() < .2
    moved = restored.geometry("curve")
    moved.P[2, :] += 1.
    restored.update("curve", geometry=moved)
    restored.savefig(tmp_path / "frame002.png", width=800, height=600)
    assert ImageChops.difference(Image.open(second), Image.open(tmp_path / "frame002.png")).getbbox()


@pytest.mark.skipif(os.environ.get("NURBSPY_TEST_PNG") != "1", reason="Explicit headless browser integration check")
def test_real_batch_frames_reuse_one_browser_and_close_the_loop(tmp_path, monkeypatch):
    from PIL import Image, ImageChops
    import nurbspy.visualization.figure_hosts as hosts

    launches, progress = [], []
    launch = hosts._launch_browser

    def counted(playwright):
        launches.append(True)
        return launch(playwright)

    monkeypatch.setattr(hosts, "_launch_browser", counted)
    figure = nrb.Figure(curve(), resolution=8, display=dict(control_points=False, control_polygon=True, control_color="#ff3300"))
    figure.set_camera(position=[3, -4, 3], target=[0, 0, 0])
    initial = figure.geometry()

    def frames():
        for displacement in (0., .5, 1., 0.):
            changed = figure.geometry()
            changed.P[:] = initial.P
            changed.P[2, 1] += displacement
            figure.update(geometry=changed)
            yield figure

    paths = figure.saveframes(frames(), tmp_path, width=400, height=300,
                              progress=lambda index, path: progress.append((index, path.name)))
    assert len(launches) == 1
    assert [index for index, _ in progress] == [1, 2, 3, 4]
    assert [path.name for path in paths] == [f"frame_{index:05d}.png" for index in range(4)]
    images = [Image.open(path).convert("RGB") for path in paths]
    assert not ImageChops.difference(images[0], images[-1]).getbbox()
    assert ImageChops.difference(images[0], images[2]).getbbox()


@pytest.mark.skipif(os.environ.get("NURBSPY_TEST_PNG") != "1", reason="Explicit headless browser integration check")
def test_batch_camera_visibility_and_appearance_updates(tmp_path):
    from PIL import Image, ImageChops

    figure = nrb.Figure(curve(), resolution=8, display=dict(control_points=False, control_polygon=False))
    figure.set_camera(position=[3, -4, 3], target=[0, 0, 0])
    camera = dict(figure.camera)

    def frames():
        yield figure
        figure.set_camera(position=[-3, -4, 3])
        yield figure  # camera-only packet, reusing geometry
        figure.update(display={"geometry_color": "#ff0000"})
        yield figure  # appearance change rebuilds layers
        figure.update(visible=False)
        yield figure  # visibility-only packet
        figure.set_camera(**camera)
        figure.update(visible=True, display={"geometry_color": "#000000"})
        yield figure

    paths = figure.saveframes(frames(), tmp_path, width=400, height=300)
    images = [Image.open(path).convert("RGB") for path in paths]
    assert len(paths) == 5
    assert not ImageChops.difference(images[0], images[-1]).getbbox()
    for index in (1, 2, 3):
        assert ImageChops.difference(images[0], images[index]).getbbox()


@pytest.mark.skipif(os.environ.get("NURBSPY_TEST_PNG") != "1", reason="Explicit headless browser integration check")
@pytest.mark.parametrize("kind", ["curve", "surface", "export_mesh"])
def test_real_png_includes_diagnostics_planar_axes_and_high_resolution(kind):
    from PIL import Image

    geometry = curve() if kind == "curve" else surface()
    display = dict(control_polygon=False, control_points=False, axes=True,
                   quantity="curvature" if kind == "curve" else "mean_curvature",
                   reverse=True, limits=[-1., 2.], colormap="coolwarm")
    if kind == "curve":
        display.update(frames=True, frame_count=4)
    if kind == "export_mesh":
        display.update(mesh="export", mesh_u=8, mesh_v=8)
    figure = nrb.Figure(geometry, resolution=12, display=display).set_camera(projection="xy")
    output = BytesIO()
    figure.savefig(output, width=2400, height=1800)
    image = Image.open(output)
    assert image.size == (2400, 1800)
    assert np.asarray(image.convert("RGB")).std() > 10


@pytest.mark.skipif(os.environ.get("NURBSPY_TEST_PNG") != "1", reason="Explicit headless browser integration check")
def test_real_editor_captures_settings_geometry_and_camera_across_object_switches():
    from playwright.sync_api import sync_playwright
    from nurbspy.visualization.figure_hosts import figure_html, _launch_browser

    figure = nrb.Figure(curve(), name="curve", resolution=8).add(surface(), name="surface")
    bridge = _FigureBridge(figure, False)
    with sync_playwright() as playwright:
        browser = _launch_browser(playwright)
        try:
            page = browser.new_page(viewport={"width": 1100, "height": 900})
            for name in ("inspect", "evaluate", "scenes", "capture"):
                page.expose_function("host_" + name, getattr(bridge, name))
            page.evaluate('''window.pywebview={api:{
                inspect:(...args)=>window.host_inspect(...args),evaluate:(...args)=>window.host_evaluate(...args),
                scenes:(...args)=>window.host_scenes(...args),capture:(...args)=>window.host_capture(...args)
            }}''')
            page.set_content(figure_html(figure, read_only=False))
            page.wait_for_function("window.nurbspyReady !== undefined")
            page.evaluate("async()=>await window.nurbspyReady")
            page.get_by_role("button", name="Open editing controls", exact=True).click()
            page.get_by_role("spinbutton", name="X coordinate", exact=True).fill("-1.5")
            page.get_by_role("spinbutton", name="X coordinate", exact=True).press("Enter")
            page.evaluate("async()=>await window.nurbspyFigure.ready()")
            first = page.evaluate("window.nurbspyFigure.snapshot()")
            assert first["objects"]["curve"]["geometry"]["parameters"]["control_points"]["values"][0][0] == -1.5
            page.get_by_role("combobox", name="Figure object", exact=True).select_option("surface")
            page.wait_for_function("window.nurbspyFigure.snapshot().active === 'surface'")
            page.evaluate("async()=>await window.nurbspyFigure.ready()")
            second = page.evaluate("window.nurbspyFigure.snapshot()")
            for key in ("position", "target", "up"):
                np.testing.assert_allclose(first["camera"][key], second["camera"][key], atol=1e-10)
            assert second["objects"]["curve"]["geometry"] == first["objects"]["curve"]["geometry"]
            page.get_by_role("button", name="Open display settings", exact=True).click()
            page.get_by_role("checkbox", name="Object visible", exact=True).uncheck()
            page.evaluate("async()=>await window.nurbspyFigure.ready()")
            assert not page.evaluate("window.nurbspyFigure.snapshot().objects.surface.visible")
            assert figure.to_state()["objects"]["surface"]["visible"]  # no shared mutation before Apply
        finally:
            browser.close()
