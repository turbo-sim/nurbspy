"""Real pointer/keyboard edits through the desktop bridge and embedded Dash host."""

import os
import threading

import numpy as np
import pytest

import nurbspy as nrb
from nurbspy.visualization.figure_hosts import (
    _FigureBridge,
    _launch_browser,
    figure_html,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("NURBSPY_TEST_PNG") != "1",
    reason="Explicit headless browser integration check",
)


@pytest.fixture(params=["desktop", "embedded"])
def editor_page(request):
    from playwright.sync_api import sync_playwright

    x, y = np.meshgrid([-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], indexing="ij")
    surface = nrb.NurbsSurface(np.stack((x, y, 0.1 * x * y)), u_degree=2, v_degree=2)
    figure = nrb.Figure(surface, name="surface", resolution=8, display={"grid": False})
    figure.add(
        nrb.NurbsCurve([nrb.Point(-1, 0, 0), nrb.Point(0, 1, 0), nrb.Point(1, 0, 0)]),
        name="curve",
    )
    figure.set_camera(
        projection="xy",
        position=[0, 0, 6],
        target=[0, 0, 0],
        up=[0, 1, 0],
        ortho_half=1.8,
    )
    server = None
    with sync_playwright() as playwright:
        browser = _launch_browser(playwright)
        page = browser.new_page(viewport={"width": 1200, "height": 950})
        errors = []
        failures = {"full": False}
        page._edit_failures = failures
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            if request.param == "desktop":
                bridge = _FigureBridge(figure, False)
                for name in ("inspect", "scenes", "capture"):
                    page.expose_function("host_" + name, getattr(bridge, name))

                def evaluate(name, packet):
                    if packet["quality"] == "full" and failures["full"]:
                        failures["full"] = False
                        return {"error": "Deliberately rejected edit"}
                    return bridge.evaluate(name, packet)

                page.expose_function("host_evaluate", evaluate)
                page.evaluate(
                    """window.pywebview={api:{
                    inspect:(...a)=>host_inspect(...a),evaluate:(...a)=>host_evaluate(...a),
                    scenes:(...a)=>host_scenes(...a),capture:(...a)=>host_capture(...a)}}"""
                )
                page.set_content(figure_html(figure, read_only=False))
                page.wait_for_function("window.nurbspyReady !== undefined")
                page.evaluate("async()=>await window.nurbspyReady")
                page.evaluate("window.testState=()=>window.nurbspyFigure.snapshot()")
                module = "nurbspy/three"
                root = "figure"
            else:
                from dash import Dash
                from werkzeug.serving import make_server
                from nurbspy.visualization import GeometryEditor

                app = Dash(__name__)
                component = GeometryEditor(figure, id="history-editor", height="850px")
                app.layout = component.layout
                component.register(app)
                server = make_server("127.0.0.1", 0, app.server, threaded=True)
                threading.Thread(target=server.serve_forever, daemon=True).start()
                page.goto(f"http://127.0.0.1:{server.server_port}")
                root = component.viewport_id
                page.locator(f"#{root} canvas.nrb-renderer").wait_for()
                page.evaluate(
                    """async root=>{const m=await import('/_nurbspy_web/figure.js');
                    window.testState=()=>m.figureState(root);}""",
                    root,
                )
                module = "/_nurbspy_web/vendor/three.module.js"

                def intercept(route):
                    if (
                        route.request.post_data_json["packet"]["quality"] == "full"
                        and failures["full"]
                    ):
                        failures["full"] = False
                        route.fulfill(
                            status=422, json={"error": "Deliberately rejected edit"}
                        )
                    else:
                        route.continue_()

                page.route("**/_nurbspy_web/figure/history-editor/edit", intercept)
            # Derive screen positions from the saved camera; exercise real pointer
            # picking rather than accessing private editor instances in the test.
            page.evaluate(
                """async({module,root})=>{
                window.testRoot=root;const THREE=await import(module);
                const update=THREE.Camera.prototype.updateMatrixWorld;
                THREE.Camera.prototype.updateMatrixWorld=function(force){
                    if(!this.userData.testProjection)window.testRenderingCamera=this;
                    return update.call(this,force);
                };
                window.testCameraFrame=()=>{
                    const c=window.testRenderingCamera,r=document.querySelector('#'+root+' canvas.nrb-renderer').getBoundingClientRect();
                    return {position:c.position.toArray(),up:c.up.toArray(),quaternion:c.quaternion.toArray(),
                        zoom:[c.zoom],projection:c.projectionMatrix.toArray(),
                        canvas:[r.x,r.y,r.width,r.height]};
                };
                window.testPositions=()=>{
                    const s=testState(),r=document.querySelector('#'+root+' canvas.nrb-renderer').getBoundingClientRect();
                    const aspect=s.rendering.width/s.rendering.height,w=Math.min(r.width,r.height*aspect),h=w/aspect;
                    const c=s.camera,half=c.ortho_half;
                    const camera=c.projection==='3d'?new THREE.PerspectiveCamera(c.fov,aspect,c.near,c.far):
                        new THREE.OrthographicCamera(-half*aspect,half*aspect,half,-half,c.near,c.far);
                    camera.userData.testProjection=true;
                    camera.position.fromArray(c.position);camera.up.fromArray(c.up);camera.lookAt(...c.target);
                    camera.zoom=c.zoom;camera.updateProjectionMatrix();camera.updateMatrixWorld();
                    const raw=s.objects[s.active].geometry.parameters.control_points.values;
                    const coords=Array.isArray(raw[0][0])?raw.map(a=>a.flat()):raw;
                    return coords[0].map((_,i)=>{const p=new THREE.Vector3(...coords.map(a=>a[i])).project(camera);
                        return [r.left+(r.width-w)/2+(p.x+1)*w/2,r.top+(r.height-h)/2+(1-p.y)*h/2];});
                };
            }""",
                dict(module=module, root=root),
            )
            wait_ready(page)
            yield page
            assert not errors
        finally:
            browser.close()
            if server:
                server.shutdown()


def wait_ready(page):
    page.wait_for_function(
        """()=>{try{testState();return !document.getElementById(testRoot).classList.contains('nrb-updating');}catch{return false;}}"""
    )
    page.evaluate(
        "async()=>{await new Promise(requestAnimationFrame);await new Promise(requestAnimationFrame);}"
    )


def points(page, name="surface"):
    return np.array(
        page.evaluate(
            "name=>testState().objects[name].geometry.parameters.control_points.values",
            name,
        )
    )


def selected(page):
    return page.evaluate(
        "JSON.parse(document.getElementById(testRoot).dataset.selectedPoints||'[0]')"
    )


def history_button(page, redo=False):
    return page.get_by_role(
        "button",
        name="Redo geometry edit" if redo else "Undo geometry edit",
        exact=True,
    )


def test_group_selection_drag_history_and_numeric_edits(editor_page, tmp_path):
    page = editor_page
    before = points(page)
    positions = page.evaluate("testPositions()")
    page.keyboard.down("Shift")
    page.mouse.click(*positions[1])
    page.keyboard.up("Shift")
    assert set(selected(page)) == {0, 1}
    assert history_button(page).is_disabled()  # Selection is not an edit.
    page.mouse.move(*positions[1])
    page.mouse.down()
    page.mouse.move(positions[1][0] + 32, positions[1][1] - 22, steps=5)
    page.mouse.up()
    wait_ready(page)
    moved = points(page)
    delta = moved - before
    np.testing.assert_allclose(delta[:, 0, 0], delta[:, 0, 1], atol=1e-12)
    np.testing.assert_allclose(delta[:, 1:, :], 0, atol=1e-12)
    np.testing.assert_allclose(delta[2], 0, atol=1e-12)  # XY view preserves native Z.
    assert np.linalg.norm(delta) > 0.05
    assert history_button(page).is_enabled()
    page.keyboard.press("Control+z")
    wait_ready(page)
    np.testing.assert_allclose(points(page), before)
    assert history_button(page).is_disabled()  # All five preview moves were one step.
    page.keyboard.press("Control+Shift+z")
    wait_ready(page)
    np.testing.assert_allclose(points(page), moved)

    # Shift rectangle adds the third point in the first control-net row.
    positions = page.evaluate("testPositions()")
    x, y = positions[2]
    page.keyboard.down("Shift")
    page.mouse.move(x - 14, y - 14)
    page.mouse.down()
    page.mouse.move(x + 14, y + 14, steps=4)
    page.mouse.up()
    page.keyboard.up("Shift")
    assert set(selected(page)) == {0, 1, 2}
    page.get_by_role("button", name="Open editing controls", exact=True).click()
    x_input = page.get_by_role("spinbutton", name="X coordinate", exact=True)
    x_input.fill("-0.4")
    x_input.press("Enter")
    wait_ready(page)
    active_edit = points(page)
    changed = np.linalg.norm(active_edit - moved, axis=0) > 1e-12
    assert (
        changed.sum() == 1 and changed[0, 2]
    )  # Numeric fields affect only the active point.
    page.screenshot(path=str(tmp_path / "editor_multiselection.png"))
    page.get_by_role("button", name="Reset", exact=True).click()
    wait_ready(page)
    np.testing.assert_allclose(points(page), before)
    history_button(page).click()
    wait_ready(page)
    np.testing.assert_allclose(points(page), active_edit)

    slider = page.get_by_role("slider", name="Logarithmic weight slider", exact=True)
    slider.focus()
    slider.press("ArrowRight")
    slider.press("Tab")
    wait_ready(page)
    weight = page.evaluate(
        "testState().objects.surface.geometry.parameters.weights.values[0][2]"
    )
    assert weight != 1
    history_button(page).click()
    wait_ready(page)
    assert (
        page.evaluate(
            "testState().objects.surface.geometry.parameters.weights.values[0][2]"
        )
        == 1
    )
    assert history_button(page, True).is_enabled()

    # A new committed edit after undo clears the redo branch.
    x_input.fill("-0.3")
    x_input.press("Enter")
    wait_ready(page)
    assert history_button(page, True).is_disabled()
    # The explicit tool replaces selection, including after an empty selection.
    page.get_by_role("button", name="Box select", exact=True).click()
    x, y = page.evaluate("testPositions()[8]")
    page.mouse.move(x - 12, y - 12)
    page.mouse.down()
    page.mouse.move(x + 12, y + 12, steps=3)
    page.mouse.up()
    assert selected(page) == [8]
    bounds = page.locator("canvas.nrb-renderer").bounding_box()
    page.mouse.click(bounds["x"] + 8, bounds["y"] + 8)
    assert selected(page) == []
    dropdown = page.get_by_role("combobox", name="Selected control point", exact=True)
    assert dropdown.is_enabled()
    dropdown.select_option("4")
    assert selected(page) == [4]


def test_history_survives_object_switch_and_shift_box_preserves_camera(editor_page):
    page = editor_page
    surface_before = points(page)
    page.get_by_role("button", name="Open editing controls", exact=True).click()
    coordinate = page.get_by_role("spinbutton", name="X coordinate", exact=True)
    coordinate.fill("-1.2")
    coordinate.press("Enter")
    wait_ready(page)
    surface_edit = points(page)
    page.get_by_role("combobox", name="Figure object", exact=True).select_option(
        "curve"
    )
    page.wait_for_function("testState().active==='curve'")
    wait_ready(page)
    assert history_button(page).is_enabled()
    page.get_by_role("button", name="Open editing controls", exact=True).click()
    curve_before = points(page, "curve")
    coordinate.fill("-1.3")
    coordinate.press("Enter")
    wait_ready(page)
    curve_edit = points(page, "curve")
    history_button(page).click()
    wait_ready(page)
    np.testing.assert_allclose(points(page, "curve"), curve_before)
    history_button(page).click()
    wait_ready(page)  # Switches back to the object being undone.
    assert page.evaluate("testState().active") == "surface"
    np.testing.assert_allclose(points(page), surface_before)
    history_button(page, True).click()
    wait_ready(page)
    np.testing.assert_allclose(points(page), surface_edit)
    history_button(page, True).click()
    wait_ready(page)
    assert page.evaluate("testState().active") == "curve"
    np.testing.assert_allclose(points(page, "curve"), curve_edit)

    camera = page.evaluate("testState().camera")
    positions = page.evaluate("testPositions()")
    x0, y0 = np.min(positions, axis=0) - 16
    x1, y1 = np.max(positions, axis=0) + 16
    page.keyboard.down("Shift")
    page.mouse.move(x0, y0)
    page.mouse.down()
    page.mouse.move(x1, y1, steps=5)
    page.mouse.up()
    page.keyboard.up("Shift")
    assert set(selected(page)) == {0, 1, 2}
    assert page.evaluate("testState().camera") == camera
    # macOS Command shortcuts work without changing the Shift selection modifier.
    page.locator("canvas.nrb-renderer").focus()
    page.keyboard.press("Meta+z")
    wait_ready(page)
    np.testing.assert_allclose(points(page, "curve"), curve_before)


def test_group_axis_constraint_cancellation_and_rejected_full_edit(editor_page):
    page = editor_page
    page.get_by_role("button", name="Open display settings", exact=True).click()
    page.get_by_role("combobox", name="Projection", exact=True).select_option("3d")
    page.get_by_role("combobox", name="Navigation mode", exact=True).select_option(
        "orbit"
    )
    wait_ready(page)
    page.get_by_role("button", name="Open editing controls", exact=True).click()
    page.get_by_role("combobox", name="Drag constraint", exact=True).select_option("x")
    positions = page.evaluate("testPositions()")
    page.keyboard.down("Shift")
    page.mouse.click(*positions[1])
    page.keyboard.up("Shift")
    assert set(selected(page)) == {0, 1}
    before = points(page)

    def drag():
        x, y = page.evaluate("testPositions()[1]")
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(x + 40, y - 20, steps=6)

    drag()
    page.mouse.up()
    wait_ready(page)
    delta = points(page) - before
    assert abs(delta[0, 0, 0]) > 0.01
    np.testing.assert_allclose(delta[:, 0, 0], delta[:, 0, 1], atol=1e-12)
    np.testing.assert_allclose(delta[1:], 0, atol=1e-12)
    history_button(page).click()
    wait_ready(page)
    np.testing.assert_allclose(points(page), before)
    assert history_button(page).is_disabled()
    # Escape cancels an in-progress drag, including already accepted previews.
    drag()
    page.keyboard.press("Escape")
    page.mouse.up()
    wait_ready(page)
    np.testing.assert_allclose(points(page), before)
    assert history_button(page).is_disabled()
    # A failed final evaluation rolls back accepted previews without recording an edit.
    drag()
    page._edit_failures["full"] = True
    page.mouse.up()
    wait_ready(page)
    np.testing.assert_allclose(points(page), before)
    assert history_button(page).is_disabled()
    page.get_by_role("combobox", name="Drag constraint", exact=True).select_option(
        "normal"
    )
    page.wait_for_function(
        "document.querySelector('.nrb-status').textContent==='Normal direction ready.'"
    )
    drag()
    page.mouse.up()
    wait_ready(page)
    normal_delta = points(page) - before
    np.testing.assert_allclose(normal_delta[:, 0, 0], normal_delta[:, 0, 1], atol=1e-12)
    assert np.linalg.norm(normal_delta[:, 0, 0]) > 0.01
    np.testing.assert_allclose(
        np.cross(normal_delta[:, 0, 0], [0, 0.1, 1]), 0, atol=1e-10
    )


@pytest.mark.parametrize(
    "projection,navigation",
    [("3d", "arcball"), ("3d", "orbit"), ("3d", "trackball"), ("xy", "orbit")],
)
@pytest.mark.parametrize("kind", ["curve", "surface"])
def test_point_drag_keeps_camera_and_viewport_fixed(
    editor_page, projection, navigation, kind
):
    page = editor_page
    if kind == "curve":
        page.get_by_role("combobox", name="Figure object", exact=True).select_option(
            kind
        )
        page.wait_for_function("testState().active==='curve'")
        wait_ready(page)
    page.get_by_role("button", name="Open display settings", exact=True).click()
    page.get_by_role("combobox", name="Projection", exact=True).select_option(
        projection
    )
    if projection == "3d":
        page.get_by_role("combobox", name="Navigation mode", exact=True).select_option(
            navigation
        )
    page.get_by_role("button", name="Open editing controls", exact=True).click()
    wait_ready(page)
    original = points(page, kind)
    baseline = page.evaluate("testCameraFrame()")
    x, y = page.evaluate("testPositions()[1]")
    page.mouse.move(x, y)
    page.mouse.down()
    for step in range(1, 5):
        page.mouse.move(x + step * 9, y - step * 6)
        # Allow both the preview reply and ResizeObserver/render callbacks to run.
        page.wait_for_timeout(60)
        frame = page.evaluate("testCameraFrame()")
        for key in baseline:
            np.testing.assert_allclose(
                frame[key],
                baseline[key],
                atol=1e-10,
                rtol=0,
                err_msg=f"{navigation} {key} changed during point drag",
            )
    page.mouse.up()
    wait_ready(page)
    frame = page.evaluate("testCameraFrame()")
    for key in baseline:
        np.testing.assert_allclose(
            frame[key],
            baseline[key],
            atol=1e-10,
            rtol=0,
            err_msg=f"{navigation} {key} changed after point drag",
        )
    assert np.linalg.norm(points(page, kind) - original) > 0.01
    # Background navigation must still work after releasing the edited point.
    bounds = page.locator("canvas.nrb-renderer").bounding_box()
    page.mouse.move(
        bounds["x"] + bounds["width"] * 0.8, bounds["y"] + bounds["height"] * 0.2
    )
    page.mouse.down(button="right")
    page.mouse.move(
        bounds["x"] + bounds["width"] * 0.8 + 30,
        bounds["y"] + bounds["height"] * 0.2 + 15,
        steps=4,
    )
    page.mouse.up(button="right")
    wait_ready(page)
    assert not np.allclose(
        page.evaluate("testCameraFrame().position"),
        baseline["position"],
        atol=1e-10,
        rtol=0,
    )
