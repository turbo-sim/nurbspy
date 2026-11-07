"""Contracts of the desktop bridge, packaging bootstrap, and public entry points."""
import base64
import gzip
import json
import re
import sys
import subprocess
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from nurbspy.visualization import GeometryEditor, edit, show
from nurbspy.visualization.desktop import _DesktopBridge, _window_html
from nurbspy.visualization.model import _GeometryModel
from nurbspy.web_explorer import NurbsExplorer
from tests.geometry.test_web_explorer import curve, surface


class Event:
    def __init__(self):
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self


class Window:
    def __init__(self):
        self.events = SimpleNamespace(closed=Event())
        self.destroy_calls = 0

    def destroy(self):
        self.destroy_calls += 1
        for handler in self.events.closed.handlers:
            handler()


def bridge_for(geometry, read_only=False):
    model = _GeometryModel(geometry, id="test", resolution=12, preview_resolution=8)
    bridge = _DesktopBridge(model, read_only=read_only)
    bridge._window = Window()
    return bridge


@pytest.mark.parametrize("factory", [curve, surface])
@pytest.mark.parametrize("quality", ["full", "preview"])
def test_desktop_uses_the_same_native_vertices_as_the_http_adapter(factory, quality):
    bridge = bridge_for(factory())
    state = bridge._model.initial_state
    state["weights"] = (np.asarray(state["weights"]) * 1.25).tolist()
    packet = dict(revision=9, action="edit", quality=quality,
                  control_points=state["control_points"], weights=state["weights"])
    result = bridge.evaluate(packet)
    expected, _, _, _ = bridge._model._render_update(packet)
    assert result["revision"] == 9 and result["quality"] == quality
    assert base64.b64decode(result["vertices"]) == expected.tobytes()


@pytest.mark.parametrize("factory", [curve, surface])
def test_apply_returns_a_new_geometry_and_never_mutates_the_input(factory):
    original = factory()
    bridge = bridge_for(original)
    state = bridge._model.initial_state
    state["control_points"] = (np.asarray(state["control_points"]) + .37).tolist()
    assert bridge.accept(state) == {"ok": True}
    assert bridge._window.destroy_calls == 1
    assert bridge._result is not original
    np.testing.assert_allclose(bridge._result.P[:original.ndim], original.P + .37)
    np.testing.assert_array_equal(original.P, np.asarray(bridge._model.initial_state["control_points"])[:original.ndim])
    assert bridge.accept(state)["error"]
    assert bridge._window.destroy_calls == 1


def test_invalid_apply_stays_open_and_cancel_returns_no_result():
    bridge = bridge_for(curve())
    state = bridge._model.initial_state
    state["weights"][0] = -1
    assert bridge.accept(state)["error"]
    assert bridge._result is None and bridge._window.destroy_calls == 0
    bridge.cancel()
    bridge.cancel()
    assert bridge._result is None and bridge._window.destroy_calls == 1
    assert bridge.evaluate({})["error"]


def test_window_manager_close_is_cancellation():
    bridge = bridge_for(curve())
    bridge._on_closed()
    assert bridge._result is None
    assert bridge.accept(bridge._model.initial_state)["error"]


def test_view_only_bridge_does_not_accept_edits():
    bridge = bridge_for(curve(), read_only=True)
    assert bridge.evaluate({})["error"]
    assert bridge.accept(bridge._model.initial_state)["error"]
    bridge.cancel()
    assert bridge._window.destroy_calls == 1


@pytest.mark.parametrize("resolution", [12, 100])
def test_startup_is_self_contained_safe_and_below_webview2_size_limit(resolution):
    title = '</script><img src=x onerror="alert(1)">'
    model = _GeometryModel(surface(), id="test", resolution=resolution, title=title)
    html = _window_html(model, read_only=False)
    assert len(html.encode("utf-8")) < 2 * 1024 * 1024
    assert title not in html
    packed = json.loads(re.search(r'const packed=("[^"]+");', html).group(1))
    configuration = json.loads(gzip.decompress(base64.b64decode(packed)))
    assert configuration["scene"]["title"] == title
    imports = configuration["modules"]["imports"]
    assert len(imports) == 16
    for name, uri in imports.items():
        assert uri.startswith("data:text/javascript;base64,")
        source = base64.b64decode(uri.split(",", 1)[1]).decode("utf-8")
        assert not re.search(r"(?:from|import)\s*['\"]\./", source), name
    assert "http://" not in html and "https://" not in html and "file://" not in html


def install_fake_webview(monkeypatch, action):
    created = {}
    window = Window()

    def create_window(title, **kwargs):
        created.update(title=title, **kwargs)
        return window

    def start(**kwargs):
        assert kwargs == dict(http_server=False, private_mode=True)
        bridge = created["js_api"]
        action(bridge)

    monkeypatch.setitem(sys.modules, "webview", SimpleNamespace(create_window=create_window, start=start))
    return created, window


def test_blocking_desktop_entry_point_returns_only_after_accepted_close(monkeypatch):
    created, window = install_fake_webview(monkeypatch, lambda bridge: bridge.accept(bridge._model.initial_state))
    original = curve()
    result = edit(original, width=900, height=600)
    assert result is not original
    np.testing.assert_array_equal(result.P[:original.ndim], original.P)
    assert window.destroy_calls == 1
    assert "html" in created and "url" not in created
    assert (created["width"], created["height"]) == (900, 600)


def test_show_is_read_only_and_returns_none(monkeypatch):
    created, window = install_fake_webview(monkeypatch, lambda bridge: bridge.cancel())
    assert show(surface()) is None
    assert created["js_api"]._read_only and window.destroy_calls == 1


def test_missing_optional_desktop_dependency_has_an_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "webview", None)
    with pytest.raises(ImportError, match="desktop extra"):
        edit(curve())


def test_desktop_rejects_non_main_thread_before_starting_gui():
    errors = []

    def invoke():
        try:
            show(curve())
        except RuntimeError as error:
            errors.append(str(error))

    thread = threading.Thread(target=invoke)
    thread.start()
    thread.join()
    assert len(errors) == 1 and "main thread" in errors[0]


@pytest.mark.parametrize("factory", [curve, surface])
def test_geometry_methods_are_thin_optional_entry_points(monkeypatch, factory):
    import nurbspy.visualization as visualization
    original = factory()
    calls = []
    monkeypatch.setattr(visualization, "show", lambda geometry, **options: calls.append((geometry, options)))
    marker = object()
    monkeypatch.setattr(visualization, "edit", lambda geometry, **options: marker)
    original.show(title="Inspect")
    assert calls == [(original, {"title": "Inspect"})]
    assert original.edit() is marker


def test_legacy_embedded_import_is_a_compatibility_alias():
    assert NurbsExplorer is GeometryEditor


def test_numerical_imports_need_no_visualization_frameworks():
    script = r'''
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'webview','dash','flask','plotly','jax','numba'}:
            raise ModuleNotFoundError('Blocked optional dependency: '+fullname, name=fullname)
sys.meta_path.insert(0,BlockOptional())
import numpy as np
import nurbspy as nrb
from nurbspy.visualization import GeometryEditor
curve=nrb.NurbsCurve(np.array([[0.,1.,2.],[0.,1.,0.]]))
assert np.isfinite(curve.get_value(.5)).all()
assert GeometryEditor(curve,id='test',resolution=8).initial_state['type']=='NurbsCurve'
try:
    curve.show()
except ImportError as error:
    assert 'desktop extra' in str(error)
else:
    raise AssertionError('Expected optional desktop dependency error')
print('base numerical imports remain independent')
'''
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run([sys.executable, "-c", script], cwd=root,
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
