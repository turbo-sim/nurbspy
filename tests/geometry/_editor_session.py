"""Shared real-browser fixture for desktop and embedded geometry interactions."""

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
    host = request.param.removesuffix("_auto")
    figure = nrb.Figure(
        surface,
        name="surface",
        resolution=None if request.param.endswith("_auto") else 8,
        display={"grid": False},
    )
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
            if host == "desktop":
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
