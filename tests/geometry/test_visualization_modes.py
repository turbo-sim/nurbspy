"""Viewer/editor capabilities use one model and adapter implementation."""

from copy import deepcopy
import pytest
from nurbspy.visualization import GeometryView, GeometryEditor
from tests.geometry.test_web_explorer import curve
from tests.geometry.test_web_explorer_transport import app_for, packet


def test_editor_extends_the_viewer_model_and_layout_contract():
    view = GeometryView(curve(3), id="view")
    editor = GeometryEditor(curve(3), id="edit")
    assert isinstance(editor, GeometryView)
    assert view.read_only and not editor.read_only
    assert view.initial_state == editor.initial_state
    assert view.edit_id not in [child.id for child in view.layout.children]
    assert editor.edit_id in [child.id for child in editor.layout.children]
    scene = next(
        child.data for child in view.layout.children if child.id == view.scene_id
    )
    assert "edit_store" not in scene


@pytest.mark.parametrize("prefix", ["/", "/host/"])
def test_embedded_viewer_has_inspection_but_no_edit_endpoint(prefix):
    view = GeometryView(curve(3), id="geometry", resolution=12)
    original = deepcopy(view.initial_state)
    app = app_for(view, prefix)
    client = app.server.test_client()
    query = packet(view)
    query.pop("action")
    query["field"] = "curvature"
    response = client.post(prefix + "_nurbspy_web/inspect/geometry", json=query)
    assert response.status_code == 200
    assert len(response.get_json()["field"]["values"]) == 60
    assert client.post(
        prefix + "_nurbspy_web/edit/geometry", json=packet(view)
    ).status_code in (404, 405)
    assert view.initial_state == original


def test_viewer_and_editor_can_share_one_host_without_duplicate_assets():
    from dash import Dash, html

    view = GeometryView(curve(3), id="view", resolution=12)
    editor = GeometryEditor(curve(3), id="edit", resolution=12)
    app = Dash(__name__)
    app.layout = html.Div([view.layout, editor.layout])
    view.register(app)
    editor.register(app)
    rules = [str(rule) for rule in app.server.url_map.iter_rules()]
    assert rules.count("/_nurbspy_web/<path:filename>") == 1
    assert "/_nurbspy_web/edit/edit" in rules and "/_nurbspy_web/edit/view" not in rules
    assert (
        app.server.test_client()
        .post("/_nurbspy_web/edit/edit", json=packet(editor))
        .status_code
        == 200
    )
