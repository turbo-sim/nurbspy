"""Viewer/edit return values compose with main's Point and YAML interfaces."""
from io import StringIO
import numpy as np
import pytest

from nurbspy import Point
from nurbspy.visualization import GeometryEditor, GeometryView
from nurbspy.visualization.desktop import _DesktopBridge


def test_point_curve_load_edit_save_and_backend_restore(curve_backend):
    original = curve_backend.NurbsCurve([Point(0, 0), Point(0.5, 1), Point(1, 0)])
    before = original.to_yaml()
    loaded = curve_backend.NurbsCurve.from_yaml(StringIO(before))
    editor = GeometryEditor(loaded, id="curve-io", resolution=12)
    state = editor.initial_state
    state["control_points"][2][1] = .25
    state["weights"][1] = 1.5
    edited = editor.geometry_from_state(state)
    restored = curve_backend.NurbsCurve.from_yaml(StringIO(edited.to_yaml()))
    assert restored.P.shape == (3, 3)
    np.testing.assert_allclose(restored.P, edited.P)
    np.testing.assert_allclose(restored.W, edited.W)
    np.testing.assert_allclose(restored.get_value(.37), edited.get_value(.37))
    assert original.to_yaml() == before


def test_point_surface_editor_and_external_stl_compose(surface_backend):
    geometry = surface_backend.NurbsSurface([
        [Point(0, 0, 0), Point(0, 1, 0)],
        [Point(1, 0, 0), Point(1, 1, .2)],
    ])
    editor = GeometryEditor(geometry, id="surface-io", resolution=12)
    state = editor.initial_state
    state["control_points"][2][1][1] = .7
    restored = surface_backend.NurbsSurface.from_yaml(StringIO(editor.geometry_from_state(state).to_yaml()))
    np.testing.assert_allclose(restored.P[2, 1, 1], .7)
    preview = editor.inspect(dict(revision=1, control_points=state["control_points"], weights=state["weights"], export_mesh=True, nu=5, nv=6))
    assert preview["mesh"]["stl"] and preview["mesh"]["triangles"]
    np.testing.assert_allclose(np.max(np.asarray(preview["mesh"]["vertices"])[:, 2]), .7, atol=1e-7)


@pytest.mark.parametrize("method", ["show", "edit"])
def test_jax_methods_use_the_shared_visualization_entry_points(jax_backend, monkeypatch, method):
    import nurbspy.visualization as visualization
    geometry = jax_backend.NurbsCurve([Point(0, 0), Point(1, 0), Point(2, 1)])
    calls = []
    monkeypatch.setattr(visualization, method, lambda value, **options: calls.append((value, options)))
    getattr(geometry, method)(title="Backend-independent UI")
    assert calls == [(geometry, {"title": "Backend-independent UI"})]
