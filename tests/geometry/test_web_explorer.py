"""Backend contract of the embeddable, directly draggable web viewport."""
from copy import deepcopy
import json

import numpy as np
import pytest

import nurbspy.numpy as nrb
from nurbspy.web_explorer import NurbsExplorer


def curve(dimension=2):
    points = np.array([[0, 1, 2, 3], [0, 2, -1, 0], [0, .2, .6, 0]], dtype=float)
    return nrb.NurbsCurve(points[:dimension], weights=np.array([1, 1.2, .8, 1]))


def surface():
    x, y = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 4), indexing="ij")
    return nrb.NurbsSurface(np.stack((x, y, x * y)), weights=np.ones(x.shape))


def request(model, revision=1, action="edit"):
    return {"revision": revision, "model": model, "action": action}


def callback_request(app, explorer, edit, state, prefix="/"):
    key = next(key for key in app.callback_map if explorer.scene_id in key)
    outputs = [{"id": output.component_id, "property": output.component_property}
               for output in app.callback_map[key]["output"]]
    return app.server.test_client().post(prefix + "_dash-update-component", json={
        "output": key, "outputs": outputs,
        "inputs": [{"id": explorer.edit_id, "property": "data", "value": edit}],
        "state": [{"id": explorer.state_id, "property": "data", "value": state}],
        "changedPropIds": [explorer.edit_id + ".data"],
    })


@pytest.mark.parametrize("factory", [curve, lambda: curve(3), surface])
def test_initial_models_are_independent_and_scene_is_finite_json(factory):
    geometry = factory()
    before = geometry.P.copy()
    explorer = NurbsExplorer(geometry, id="geometry", resolution=8)
    scene = explorer.evaluate_scene()
    json.dumps(scene, allow_nan=False)
    assert scene["control_shape"] == list(geometry.W.shape)
    assert np.shape(scene["control_points"]) == (geometry.W.size, 3)
    assert len(scene["points"]) == (64 if explorer.surface else 40)
    assert bool(scene["triangles"]) == explorer.surface
    state = explorer.initial_state
    state["control_points"][0][0] = 17 if not explorer.surface else [17] * 4
    np.testing.assert_array_equal(geometry.P, before)
    np.testing.assert_array_equal(explorer.geometry_from_state().P[:len(before)], before)


def test_curve_drag_and_weight_edit_change_native_geometry_without_mutation():
    original = curve()
    explorer = NurbsExplorer(original, id="curve", resolution=8)
    state = explorer.initial_state
    baseline = deepcopy(state)
    edit = deepcopy(state)
    edit["control_points"][1][1] += 1.1
    edit["weights"][1] = 2.7
    scene, accepted = explorer.process_edit(request(edit, revision=12), state)
    assert scene["revision"] == 12 and scene["error"] is None
    assert accepted == edit
    assert not np.allclose(scene["points"], explorer.evaluate_scene()["points"])
    geometry = explorer.geometry_from_state(accepted)
    assert geometry.P[1, 1] == pytest.approx(original.P[1, 1] + 1.1)
    assert geometry.W[1] == pytest.approx(2.7)
    assert state == baseline == explorer.initial_state
    np.testing.assert_array_equal(original.P, np.asarray(baseline["control_points"])[:original.ndim])


def test_surface_control_flattening_matches_u_then_v_indexing():
    geometry = surface()
    explorer = NurbsExplorer(geometry, id="surface", resolution=8)
    state = explorer.initial_state
    state["control_points"][2][1][2] = 3
    scene, accepted = explorer.process_edit(request(state))
    assert scene["control_points"][1 * 4 + 2][2] == 3
    assert scene["control_lines"] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11],
                                       [0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]]
    assert len(scene["triangles"]) == 2 * 7 * 7
    assert max(max(face) for face in scene["triangles"]) == 63
    np.testing.assert_allclose(explorer.geometry_from_state(accepted).get_value(.5, .5).ravel(),
                               nrb.NurbsSurface(np.asarray(state["control_points"])).get_value(.5, .5).ravel())


def test_reset_and_two_browser_states_do_not_interfere():
    explorer = NurbsExplorer(curve(), id="geometry", resolution=8)
    one, two = explorer.initial_state, explorer.initial_state
    one["control_points"][0][0] = 10
    _, accepted = explorer.process_edit(request(one), two)
    reset_scene, reset_model = explorer.process_edit(request(None, 2, "reset"), accepted)
    assert reset_scene["error"] is None
    assert reset_model == two == explorer.initial_state
    assert accepted == one and accepted != two


@pytest.mark.parametrize("mutation", [
    lambda state: state["control_points"][0].append(5),
    lambda state: state["control_points"][0].__setitem__(0, float("nan")),
    lambda state: state["control_points"][0].__setitem__(0, 1j),
    lambda state: state["control_points"][0].__setitem__(0, "3"),
    lambda state: state["weights"].__setitem__(0, 0),
    lambda state: state["weights"].__setitem__(0, -1),
    lambda state: state["weights"].__setitem__(0, float("inf")),
    lambda state: state.__setitem__("degree", 1),
    lambda state: state["knots"].__setitem__(0, -1),
    lambda state: state.__setitem__("type", "NurbsSurface"),
    lambda state: state.__setitem__("extra", "unexpected"),
])
def test_invalid_edit_is_atomic_and_acknowledges_revision(mutation):
    explorer = NurbsExplorer(curve(), id="curve", resolution=8)
    state, edit = explorer.initial_state, explorer.initial_state
    mutation(edit)
    scene, model = explorer.process_edit(request(edit, 17), state)
    assert scene["revision"] == 17 and scene["error"]
    assert model == state == explorer.initial_state
    json.dumps(scene, allow_nan=False)


@pytest.mark.parametrize("edit", [None, {}, {"revision": True}, {"revision": -1},
                                  {"revision": 1.5}, {"revision": 1, "action": "delete"}])
def test_invalid_protocol_does_not_crash(edit):
    explorer = NurbsExplorer(curve(), id="curve", resolution=8)
    scene, model = explorer.process_edit(edit)
    assert scene["error"] and model == explorer.initial_state


def test_invalid_store_falls_back_to_original():
    explorer = NurbsExplorer(curve(), id="curve", resolution=8)
    scene, accepted = explorer.process_edit(request(None), {"not": "a valid state"})
    assert scene["error"] and accepted == explorer.initial_state


@pytest.mark.parametrize("identifier", ["", "3curve", "with spaces", "bad/path", None])
def test_ids_are_explicit_and_safe(identifier):
    with pytest.raises(ValueError, match="id"):
        NurbsExplorer(curve(), id=identifier)


@pytest.mark.parametrize("resolution", [True, 7, 101, 12.5])
def test_resolution_validation(resolution):
    with pytest.raises(ValueError, match="resolution"):
        NurbsExplorer(curve(), id="curve", resolution=resolution)


def test_unsupported_input_geometry_fails_clearly():
    with pytest.raises(TypeError, match="nurbspy"):
        NurbsExplorer(object(), id="bad")
    with pytest.raises(ValueError, match="two or three"):
        NurbsExplorer(nrb.NurbsCurve(np.ones((1, 4))), id="bad")
    bad = curve()
    bad.P = bad.P.astype(complex)
    with pytest.raises(ValueError, match="real"):
        NurbsExplorer(bad, id="bad")


@pytest.mark.parametrize("prefix", ["/", "/host/"])
def test_dash_callbacks_register_without_owning_app_and_respect_url_prefix(prefix):
    dash = pytest.importorskip("dash")
    app = dash.Dash(__name__, url_base_pathname=prefix)
    first = NurbsExplorer(curve(), id="curve", resolution=8)
    second = NurbsExplorer(surface(), id="surface", resolution=8)
    app.layout = dash.html.Div([first.layout, second.layout])
    first.register(app)
    second.register(app)
    assert len(app.callback_map) == 4
    with pytest.raises(ValueError, match="already registered"):
        first.register(app)
    client = app.server.test_client()
    assert client.get(prefix).status_code == 200
    assert client.get(prefix + "_dash-layout").status_code == 200
    dependencies = client.get(prefix + "_dash-dependencies").get_json()
    assert len(dependencies) == 4
    # Static asset is served by the host Flask app. It may be written by the
    # frontend task after this backend test is first run.
    from pathlib import Path
    asset = Path(__file__).resolve().parents[2] / "src/nurbspy/web_assets/explorer.js"
    if asset.exists():
        response = client.get(prefix + "_nurbspy_web/explorer.js")
        assert response.status_code == 200
        assert b"export" in response.data
    state = first.initial_state
    edit = deepcopy(state)
    edit["control_points"][1][1] += .75
    response = callback_request(app, first, request(edit, 5), state, prefix)
    assert response.status_code == 200, response.get_data(as_text=True)
    result = response.get_json()["response"]
    assert result[first.scene_id]["data"]["revision"] == 5
    assert result[first.state_id]["data"] == edit
    assert second.initial_state["type"] == "NurbsSurface"
    assert client.get(prefix + "_nurbspy_web/../web_explorer.py").status_code == 404


def test_jax_input_is_host_copy_and_native_model_stays_differentiable():
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    from nurbspy.jax import NurbsCurve
    geometry = NurbsCurve(jnp.asarray(curve().P))
    explorer = NurbsExplorer(geometry, id="jax", resolution=8)
    state = explorer.initial_state
    state["control_points"][1][1] += 1
    explorer.process_edit(request(state))
    assert not isinstance(explorer.geometry_from_state().P, jax.Array)
    np.testing.assert_array_equal(geometry.P, curve().P)
    derivative = jax.jacfwd(lambda parameters: NurbsCurve(parameters).get_value(.5))(geometry.P)
    assert np.all(np.isfinite(derivative))
