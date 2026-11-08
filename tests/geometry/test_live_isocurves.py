"""Exact surface-linked isocurves across editing, persistence, and export."""

from io import StringIO
import os

import numpy as np
import pytest
import nurbspy as nrb
from nurbspy.visualization.model import _GeometryModel
from tests.geometry._editor_session import editor_page, wait_ready, history_button


def surface():
    x, y = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 4), indexing="ij")
    points = np.stack((x, y, 0.3 * x * y + 0.2 * np.sin(x)))
    weights = 1 + 0.3 * np.arange(20).reshape(5, 4) / 20
    return nrb.NurbsSurface(points, weights=weights, u_degree=2, v_degree=2)


@pytest.mark.parametrize("resolution", [None, 12])
@pytest.mark.parametrize("quality", ["preview", "full"])
def test_isocurves_match_native_curves_after_point_and_weight_edits(
    resolution, quality
):
    original = surface()
    model = _GeometryModel(
        original,
        id="iso",
        resolution=resolution,
        display=dict(isocurves=True, isocurves_u=[0.17, 0.83], isocurves_v=[0.39]),
    )
    state = model.initial_state
    state["control_points"][2][1][2] += 1.2
    state["weights"][0][1] = 2.5
    request = dict(
        revision=3,
        quality=quality,
        isocurves=True,
        control_points=state["control_points"],
        weights=state["weights"],
    )
    result = model.inspect(request)
    geometry = model.geometry_from_state(state)
    sampling, _ = model._request_sampling(geometry, request)
    lines = result["isocurves"]["curves"]
    assert [(line["direction"], line["parameter"]) for line in lines] == [
        ("u", 0.17),
        ("u", 0.83),
        ("v", 0.39),
    ]
    for line in lines:
        fixed = line["parameter"]
        axis = np.unique(sampling[quality][1 if line["direction"] == "u" else 0])
        curve = getattr(geometry, "get_isocurve_" + line["direction"])(fixed)
        np.testing.assert_allclose(
            np.asarray(line["points"]).T, curve.get_value(axis), atol=1e-13
        )
        if quality == "preview" and resolution is None:
            assert len(line["points"]) <= 20
    np.testing.assert_array_equal(original.P, surface().P)
    assert "isocurves" not in model.inspect({**request, "isocurves": False})


def test_native_parameter_domains_one_direction_and_saved_state():
    geometry = surface()
    geometry.U = geometry.U * 3 + 2
    geometry.V = geometry.V * 7 - 3
    figure = nrb.Figure(
        geometry,
        display=dict(
            isocurves=True,
            isocurves_u=[2.7, 4.1],
            isocurves_v=[],
            isocurves_color="#9020ab",
        ),
    )
    restored = nrb.Figure.from_yaml(StringIO(figure.to_yaml()))
    assert restored.to_state() == figure.to_state()
    assert restored.names == ("geometry",)
    lines = restored._payload()["scenes"]["geometry"]["diagnostics"]["isocurves"][
        "curves"
    ]
    assert [line["parameter"] for line in lines] == [2.7, 4.1]
    default = nrb.Figure(geometry)._payload()["scenes"]["geometry"]["diagnostics"][
        "isocurves"
    ]["curves"]
    assert [line["parameter"] for line in default[:5]] == pytest.approx(
        np.linspace(2.24, 4.76, 5)
    )
    state = figure.to_state()
    for key in ("isocurves", "isocurves_u", "isocurves_v", "isocurves_color"):
        state["objects"]["geometry"]["display"].pop(key)
    assert nrb.Figure.from_state(state).settings()["isocurves"] is False


@pytest.mark.parametrize(
    "options",
    [
        {"isocurves": "yes"},
        {"isocurves_u": [True]},
        {"isocurves_u": [float("nan")]},
        {"isocurves_u": [0.7, 0.2]},
        {"isocurves_v": [0.2, 0.2]},
        {"isocurves_v": [1.1]},
        {"isocurves_u": np.linspace(0, 1, 31).tolist()},
        {"isocurves_color": "grey"},
    ],
)
def test_invalid_isocurve_settings_are_rejected(options):
    with pytest.raises(ValueError):
        nrb.Figure(surface(), display=options)


def test_curve_cannot_enable_surface_isocurves():
    curve = nrb.NurbsCurve([[0, 1, 2], [0, 1, 0], [0, 0, 0]])
    with pytest.raises(ValueError, match="require a surface"):
        nrb.Figure(curve, display={"isocurves": True})


def capture_isocurves(page):
    page.evaluate("""async()=>{
        const THREE=await import(location.protocol==='file:'||location.href==='about:blank'?'nurbspy/three':'/_nurbspy_web/vendor/three.module.js');
        const add=THREE.Object3D.prototype.add;
        THREE.Object3D.prototype.add=function(...items){
            for(const item of items)if(item.name==='Surface isocurves')window.testIsoGroup=item;
            return add.apply(this,items);
        };
        const update=THREE.Scene.prototype.updateMatrixWorld;
        THREE.Scene.prototype.updateMatrixWorld=function(...args){
            this.traverse(item=>{if(item.name==='Surface isocurves')window.testIsoGroup=item;});
            return update.apply(this,args);
        };
        window.testIsocurves=()=>window.testIsoGroup.children.map(line=>{
            const starts=line.geometry.attributes.instanceStart,ends=line.geometry.attributes.instanceEnd;
            const points=Array.from({length:starts.count},(_,i)=>[starts.getX(i),starts.getY(i),starts.getZ(i)]);
            points.push([ends.getX(ends.count-1),ends.getY(ends.count-1),ends.getZ(ends.count-1)]);
            return {...line.userData.isocurve,points};
        });
    }""")


def assert_browser_endpoints_follow_surface(page):
    page.wait_for_function(
        "()=>{try{return testIsoGroup.visible&&testState().objects.surface;}catch{return false;}}"
    )
    state = page.evaluate("testState().objects.surface.geometry.parameters")
    geometry = nrb.NurbsSurface(
        np.array(state["control_points"]["values"]),
        weights=np.array(state["weights"]["values"]),
        u_degree=2,
        v_degree=2,
    )
    lines = page.evaluate("testIsocurves()")
    assert len(lines) == 10
    for line in lines:
        ends = np.array(line["points"])[[0, -1]].T
        parameter = line["parameter"]
        expected = (
            geometry.get_value(np.full(2, parameter), np.array([0.0, 1.0]))
            if line["direction"] == "u"
            else geometry.get_value(np.array([0.0, 1.0]), np.full(2, parameter))
        )
        np.testing.assert_allclose(ends, expected, atol=2e-6)
    return lines


@pytest.mark.skipif(
    os.environ.get("NURBSPY_TEST_PNG") != "1",
    reason="Explicit browser integration check",
)
@pytest.mark.parametrize(
    "editor_page",
    ["desktop", "embedded", "desktop_auto", "embedded_auto"],
    indirect=True,
)
def test_edit_weight_drag_history_and_toggle_update_isocurves(editor_page):
    page = editor_page
    capture_isocurves(page)
    page.get_by_role("button", name="Open display settings", exact=True).click()
    page.get_by_role("checkbox", name="Isocurves", exact=True).check()
    original = assert_browser_endpoints_follow_surface(page)
    camera = page.evaluate("testState().camera")
    page.get_by_role("button", name="Open editing controls", exact=True).click()
    z = page.get_by_role("spinbutton", name="Z coordinate", exact=True)
    z.fill("1.25")
    z.press("Tab")
    wait_ready(page)
    changed = assert_browser_endpoints_follow_surface(page)
    assert changed != original
    weight = page.get_by_role("spinbutton", name="Control point weight", exact=True)
    weight.fill("2.2")
    weight.press("Tab")
    wait_ready(page)
    weighted = assert_browser_endpoints_follow_surface(page)
    assert weighted != changed
    history_button(page).click()
    wait_ready(page)
    assert assert_browser_endpoints_follow_surface(page) == changed
    history_button(page, redo=True).click()
    wait_ready(page)
    assert assert_browser_endpoints_follow_surface(page) == weighted
    x, y = page.evaluate("testPositions()[0]")
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + 25, y - 20, steps=4)
    page.wait_for_function(
        "()=>testIsoGroup.visible&&testIsoGroup.userData.quality==='preview'"
    )
    assert max(len(line["points"]) for line in page.evaluate("testIsocurves()")) <= 20
    page.mouse.up()
    wait_ready(page)
    dragged = assert_browser_endpoints_follow_surface(page)
    assert dragged != weighted
    assert page.evaluate("testState().camera") == camera
    assert page.evaluate("Object.keys(testState().objects)") == ["surface", "curve"]
    page.get_by_role("button", name="Open display settings", exact=True).click()
    toggle = page.get_by_role("checkbox", name="Isocurves", exact=True)
    toggle.uncheck()
    page.wait_for_function("()=>!testIsoGroup.visible")
    assert page.evaluate("testState().objects.surface.display.isocurves") is False
    toggle.check()
    assert_browser_endpoints_follow_surface(page)


@pytest.mark.skipif(
    os.environ.get("NURBSPY_TEST_PNG") != "1", reason="Explicit export browser check"
)
def test_isocurves_survive_yaml_png_and_offline_html(tmp_path):
    from PIL import Image
    from playwright.sync_api import sync_playwright
    from nurbspy.visualization.figure_hosts import _launch_browser

    figure = nrb.Figure(
        surface(),
        name="surface",
        display=dict(
            isocurves=True,
            isocurves_u=[0.17, 0.83],
            isocurves_v=[],
            isocurves_color="#9020ab",
            grid=True,
        ),
    )
    figure = nrb.Figure.from_yaml(StringIO(figure.to_yaml()))
    changed = figure.geometry()
    changed.P[2, 0, 0] += 1
    changed.W[0, 1] = 2
    figure.update(geometry=changed)
    png, html = tmp_path / "linked.png", tmp_path / "linked.html"
    figure.savefig(png, width=800, height=600)
    figure.to_html(html)
    without = figure.copy().update(display={"isocurves": False})
    without.savefig(tmp_path / "without.png", width=800, height=600)
    with Image.open(png) as image, Image.open(tmp_path / "without.png") as other:
        assert image.size == (800, 600)
        assert not np.array_equal(np.asarray(image), np.asarray(other))
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
        page.goto(html.as_uri())
        page.wait_for_function("window.nurbspyReportReady === true")
        page.evaluate("async()=>await window.nurbspyFigure.ready()")
        capture_isocurves(page)
        page.get_by_role("button", name="Open display settings", exact=True).click()
        page.wait_for_function("window.testIsoGroup !== undefined")
        toggle = page.get_by_role("checkbox", name="Isocurves", exact=True)
        assert toggle.is_checked()
        assert (
            page.get_by_label("Isocurve colour", exact=True).input_value() == "#9020ab"
        )
        for preset in ("fine", "fast", "auto"):
            page.get_by_role(
                "combobox", name="Display quality", exact=True
            ).select_option(preset)
            page.evaluate("async()=>await window.nurbspyFigure.ready()")
            lines = page.evaluate("testIsocurves()")
            assert [line["parameter"] for line in lines] == [0.17, 0.83]
            for line in lines:
                np.testing.assert_allclose(
                    np.array(line["points"])[[0, -1]].T,
                    changed.get_value(
                        np.full(2, line["parameter"]), np.array([0.0, 1.0])
                    ),
                    atol=2e-6,
                )
        toggle.uncheck()
        page.wait_for_function("()=>!testIsoGroup.visible")
        toggle.check()
        page.evaluate("async()=>await window.nurbspyFigure.ready()")
        assert page.evaluate("testIsoGroup.visible")
        assert not errors and not network
        browser.close()
