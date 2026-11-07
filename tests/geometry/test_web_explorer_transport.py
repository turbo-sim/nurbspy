"""The fast display protocol keeps native Python geometry authoritative."""
from copy import deepcopy

import numpy as np
import pytest

from nurbspy.web_explorer import NurbsExplorer
from tests.geometry.test_web_explorer import curve, surface


def app_for(explorer, prefix="/"):
    from dash import Dash
    app = Dash(__name__, url_base_pathname=prefix)
    app.layout = explorer.layout
    explorer.register(app)
    return app


def packet(explorer, quality="full", revision=1):
    model = explorer.initial_state
    return dict(revision=revision, action="edit", quality=quality,
                control_points=model["control_points"], weights=model["weights"])


@pytest.mark.parametrize("factory", [curve, lambda: curve(3), surface])
@pytest.mark.parametrize("quality", ["full", "preview"])
@pytest.mark.parametrize("prefix", ["/", "/host/"])
def test_binary_edit_matches_native_evaluation(factory, quality, prefix):
    original = factory()
    explorer = NurbsExplorer(original, id="geometry", resolution=35, preview_resolution=12)
    app = app_for(explorer, prefix)
    data = packet(explorer, quality, 7)
    if explorer.surface:
        data["control_points"][2][1][1] += .317
        data["weights"][1][1] = 2.125
    else:
        data["control_points"][1][1] += .317
        data["weights"][1] = 2.125
    response = app.server.test_client().post(prefix + "_nurbspy_web/edit/geometry", json=data)
    assert response.status_code == 200
    assert response.mimetype == "application/octet-stream"
    assert response.headers["X-Nurbspy-Revision"] == "7"
    assert response.headers["X-Nurbspy-Quality"] == quality
    positions = np.frombuffer(response.data, dtype="<f4").reshape(-1, 3)
    model = explorer.initial_state
    model.update(control_points=data["control_points"], weights=data["weights"])
    edited = explorer.geometry_from_state(model)
    count = 35 if quality == "full" else 12
    u = np.linspace(0, 1, count if explorer.surface else 5*count)
    if explorer.surface:
        u, v = np.meshgrid(u, u, indexing="ij")
        expected = edited.get_value(u.ravel(), v.ravel())
    else:
        expected = edited.get_value(u)
    expected = np.pad(expected, ((0, 3-explorer.dimension), (0, 0))).T.astype("<f4")
    np.testing.assert_array_equal(positions, expected)
    np.testing.assert_array_equal(original.P, np.asarray(explorer.initial_state["control_points"])[:original.ndim])


@pytest.mark.parametrize("mutate", [
    lambda data: data.update(revision=True),
    lambda data: data.update(revision=-1),
    lambda data: data.update(action="unknown"),
    lambda data: data.update(quality="ultra"),
    lambda data: data.update(quality=[]),
    lambda data: data.update(knots=[0, 1]),
    lambda data: data.update(weights=[1, 2]),
    lambda data: data["weights"].__setitem__(0, 0),
    lambda data: data["control_points"][0].__setitem__(0, float("nan")),
    lambda data: data.update(control_points=[[1], [2]]),
])
def test_invalid_compact_requests_are_rejected_without_mutation(mutate):
    explorer = NurbsExplorer(curve(), id="geometry", resolution=8)
    app = app_for(explorer)
    state = explorer.initial_state
    request = packet(explorer)
    mutate(request)
    response = app.server.test_client().post("/_nurbspy_web/edit/geometry", json=request)
    assert response.status_code == 422 and response.get_json()["error"]
    assert explorer.initial_state == state


def test_reset_and_independent_requests_are_stateless():
    explorer = NurbsExplorer(surface(), id="geometry", resolution=20, preview_resolution=8)
    data = packet(explorer)
    data["weights"][1][1] = 3
    one, model, _, _ = explorer._render_update(data)
    reset, reset_model, _, _ = explorer._render_update(dict(revision=2, action="reset"))
    assert reset_model == explorer.initial_state and model != reset_model
    assert not np.array_equal(one, reset)
    np.testing.assert_array_equal(explorer._render_update(data)[0], one)


def test_cached_topology_is_not_exposed_for_mutation():
    explorer = NurbsExplorer(surface(), id="geometry", resolution=20, preview_resolution=8)
    scene = explorer.evaluate_scene()
    assert len(scene["topologies"]["preview"]["triangles"]) == 2*7*7
    scene["triangles"][0][0] = 99999
    scene["topologies"]["full"]["triangles"][0][0] = 99999
    assert max(explorer.evaluate_scene()["triangles"][0]) < 400


@pytest.mark.parametrize("preview", [True, 0, 7, 21, 8.5])
def test_preview_resolution_errors_are_clear(preview):
    with pytest.raises(ValueError, match="preview_resolution"):
        NurbsExplorer(surface(), id="geometry", resolution=20, preview_resolution=preview)


def test_display_rounding_does_not_change_model_precision():
    explorer = NurbsExplorer(curve(), id="geometry", resolution=8)
    data = packet(explorer)
    data["control_points"][1][1] = .12345678901234567
    _, model, _, _ = explorer._render_update(data)
    assert explorer.geometry_from_state(model).P[1, 1] == data["control_points"][1][1]


def test_endpoint_does_not_send_topology_or_model_on_each_edit():
    explorer = NurbsExplorer(surface(), id="geometry", resolution=100)
    app = app_for(explorer)
    response = app.server.test_client().post("/_nurbspy_web/edit/geometry", json=packet(explorer))
    assert response.status_code == 200
    assert len(response.data) == 100*100*3*4
