"""Blocking desktop windows using a direct Python bridge, without an HTTP server."""

import base64
import gzip
from functools import lru_cache
from html import escape
import json
from pathlib import Path
import threading

from .model import _GeometryModel


def _script_json(value):
    """Embed JSON without allowing user titles to terminate an HTML script."""
    return (
        json.dumps(value, allow_nan=False)
        .replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


@lru_cache(maxsize=1)
def _module_map():
    """Bundle local ES modules as data URLs; pywebview needs no asset server.

    Import-map names replace only the known, bundled relative dependencies.
    The original sources and vendor license remain in the installed package.
    """
    assets = Path(__file__).parents[1] / "web_assets"
    modules = {
        "nurbspy/core": "vendor/three.core.js",
        "nurbspy/three": "vendor/three.module.js",
        "nurbspy/orbit": "vendor/OrbitControls.js",
        "nurbspy/trackball": "vendor/TrackballControls.js",
        "nurbspy/arcball": "vendor/ArcballControls.js",
        "nurbspy/cube": "viewcube.js",
        "nurbspy/display": "display.js",
        "nurbspy/science": "science.js",
        "nurbspy/panels": "panels.js",
        "nurbspy/editing": "editing.js",
        "nurbspy/line": "vendor/Line2.js",
        "nurbspy/line-segments": "vendor/LineSegments2.js",
        "nurbspy/line-geometry": "vendor/LineGeometry.js",
        "nurbspy/line-segments-geometry": "vendor/LineSegmentsGeometry.js",
        "nurbspy/line-material": "vendor/LineMaterial.js",
        "nurbspy/editor": "explorer.js",
        "nurbspy/figure": "figure.js",
    }
    replacements = {
        "./three.core.js": "nurbspy/core",
        "./three.module.js": "nurbspy/three",
        "./vendor/three.module.js": "nurbspy/three",
        "./vendor/OrbitControls.js": "nurbspy/orbit",
        "./vendor/TrackballControls.js": "nurbspy/trackball",
        "./vendor/ArcballControls.js": "nurbspy/arcball",
        "./viewcube.js": "nurbspy/cube",
        "./display.js": "nurbspy/display",
        "./science.js": "nurbspy/science",
        "./panels.js": "nurbspy/panels",
        "./editing.js": "nurbspy/editing",
        "./explorer.js": "nurbspy/editor",
        "./vendor/Line2.js": "nurbspy/line",
        "./Line2.js": "nurbspy/line",
        "./LineSegments2.js": "nurbspy/line-segments",
        "./vendor/LineGeometry.js": "nurbspy/line-geometry",
        "./LineGeometry.js": "nurbspy/line-geometry",
        "./LineSegmentsGeometry.js": "nurbspy/line-segments-geometry",
        "./vendor/LineMaterial.js": "nurbspy/line-material",
        "./LineMaterial.js": "nurbspy/line-material",
    }
    imports = {}
    for name, filename in modules.items():
        source = (assets / filename).read_text(encoding="utf-8")
        for relative, mapped in replacements.items():
            for quote in ("'", '"'):
                source = source.replace(
                    quote + relative + quote, quote + mapped + quote
                )
        imports[name] = "data:text/javascript;base64," + base64.b64encode(
            source.encode()
        ).decode("ascii")
    return {"imports": imports}


def _window_html(model, *, read_only):
    """Self-contained HTML, ES modules, and bridge bootstrap for one window."""
    title = escape(model.title)
    configuration = _script_json(
        {"scene": model.evaluate_scene(), "modules": _module_map()}
    )
    # WebView2 NavigateToString accepts at most 2 MiB. Compressed startup data
    # avoids that limit without introducing local file origins or an HTTP server.
    packed = base64.b64encode(
        gzip.compress(configuration.encode("utf-8"), mtime=0)
    ).decode("ascii")
    return (
        """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>"""
        + title
        + """</title><style>
html,body{margin:0;height:100%;background:#ffffff;color:#203440;font:14px system-ui,sans-serif}
body{display:flex;flex-direction:column;box-sizing:border-box;padding:16px;gap:12px}
#geometry{min-height:0;flex:1;height:auto}.window-footer{display:flex;align-items:center;justify-content:flex-end;gap:10px}
#message{flex:1;color:#536d79;font-size:12px}#message:empty{display:none}button{font:inherit;padding:9px 18px;border:1px solid #cbd7de;border-radius:7px;background:white;color:#294956;cursor:pointer}
#apply{background:var(--nrb-accent,#0072BD);color:white;border-color:var(--nrb-accent,#0072BD)}button:disabled{opacity:.45;cursor:wait}
button:focus-visible{outline:2px solid var(--nrb-accent,#0072BD);outline-offset:2px}#message.error{color:#a7342b}
</style></head><body>
<div id="geometry" role="region" aria-label="Geometry viewer"></div>
<footer class="window-footer"><span id="message" role="alert"></span>
<button id="cancel" type="button" disabled>"""
        + ("Close" if read_only else "Cancel")
        + """</button>
<button id="apply" type="button" disabled>Apply</button></footer>
<script type="module">
const packed="""
        + _script_json(packed)
        + ";\nconst readOnly="
        + str(read_only).lower()
        + """;
let update,editorState;
const apply=document.getElementById('apply'), cancel=document.getElementById('cancel'), message=document.getElementById('message');
apply.hidden=readOnly;
function fail(error){message.textContent=error.message || String(error);message.classList.add('error');}
function decode(data){
    const bytes=Uint8Array.from(atob(data),character=>character.charCodeAt(0));
    return new Float32Array(bytes.buffer);
}
async function mount(){
    cancel.disabled=false;
    try {
        if(!window.DecompressionStream) throw new Error('This viewer requires a current webview runtime with gzip decompression support.');
        const zipped=Uint8Array.from(atob(packed),character=>character.charCodeAt(0));
        const configuration=await new Response(new Blob([zipped]).stream().pipeThrough(new DecompressionStream('gzip'))).json();
        const importMap=document.createElement('script');importMap.type='importmap';
        importMap.textContent=JSON.stringify(configuration.modules);document.head.append(importMap);
        ({update,editorState}=await import('nurbspy/editor'));
        const result=update('geometry',configuration.scene,{
            readOnly,
            inspect:async packet=>{
                const result=await window.pywebview.api.inspect(packet);
                if(result.error) throw new Error(result.error);
                return result;
            },
            evaluate:async packet=>{
                const result=await window.pywebview.api.evaluate(packet);
                if(result.error) throw new Error(result.error);
                return {revision:result.revision,quality:result.quality,positions:decode(result.vertices)};
            },
            onBusyChange:busy=>{apply.disabled=busy;},
        });
        if(result.error) throw new Error(result.error);
        apply.disabled=false;
        message.textContent='';
    }catch(error){fail(error);}
}
apply.addEventListener('click',async()=>{
    try {
        const state=editorState('geometry');
        if(state.busy) return;
        apply.disabled=true;
        const result=await window.pywebview.api.accept(state.model);
        if(result.error) throw new Error(result.error);
    }catch(error){apply.disabled=false;fail(error);}
});
cancel.addEventListener('click',()=>window.pywebview.api.cancel());
window.addEventListener('keydown',event=>{
    if(event.key==='Escape' && window.pywebview?.api){event.preventDefault();window.pywebview.api.cancel();}
});
if(window.pywebview?.api) mount();
else window.addEventListener('pywebviewready',mount,{once:true});
</script></body></html>"""
    )


class _DesktopBridge:
    """Explicit evaluation, inspection, accept, and cancel bridge methods."""

    def __init__(self, model, *, read_only):
        self._model, self._read_only = model, read_only
        self._window = None
        self._result = None
        self._closed = False
        self._lock = threading.Lock()

    def evaluate(self, packet):
        """Evaluate using native nurbspy and encode display vertices for IPC."""
        try:
            if self._read_only or self._closed:
                raise ValueError("This window is not accepting edits")
            vertices, _, quality, revision = self._model._render_update(packet)
            return {
                "revision": revision,
                "quality": quality,
                "vertices": base64.b64encode(vertices.tobytes()).decode("ascii"),
            }
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            return {"error": str(error)}

    def inspect(self, packet):
        """Read-only native diagnostics are also available in show() windows."""
        try:
            if self._closed:
                raise ValueError("This window has already closed")
            return self._model.inspect(packet)
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            return {"error": str(error)}

    def accept(self, state):
        """Validate independently of the frontend, copy native geometry, close."""
        try:
            if self._read_only:
                raise ValueError("This window is view-only")
            geometry = self._model.geometry_from_state(state)
            self._model._render_update(
                dict(
                    revision=1,
                    action="edit",
                    quality="full",
                    control_points=state["control_points"],
                    weights=state["weights"],
                )
            )
            with self._lock:
                if self._closed:
                    raise ValueError("This window has already closed")
                self._result, self._closed = geometry, True
            self._window.destroy()
            return {"ok": True}
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            return {"error": str(error)}

    def cancel(self):
        """Cancel is also used by the view-only Close button."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._window.destroy()

    def _on_closed(self, *args):
        with self._lock:
            self._closed = True


def _open(
    geometry,
    *,
    read_only,
    title=None,
    resolution=40,
    preview_resolution=None,
    width=1100,
    height=780,
    display=None
):
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError(
            "Desktop visualization must be called on Python's main thread"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 400
        for value in (width, height)
    ):
        raise ValueError(
            "Window width and height must be integers of at least 400 pixels"
        )
    from .figure import Figure
    from .figure_hosts import open_figure

    if isinstance(geometry, Figure):
        figure = geometry
    else:
        figure = Figure(geometry, title=title or "NURBS geometry",
                        resolution=resolution, preview_resolution=preview_resolution,
                        display=display)
    return open_figure(figure, read_only=read_only, width=width, height=height)


def show(geometry, **kwargs):
    """Block while inspecting geometry in a desktop window; return None.

    Accepts title, resolution (8–100), preview_resolution, width, and height.
    Does not edit the input. Requires pywebview and the platform's webview runtime.
    """
    _open(geometry, read_only=True, **kwargs)


def edit(geometry, **kwargs):
    """Block while editing; return a new NumPy geometry on Apply, else None.

    Accepts the same options as show(). Original NumPy/JAX geometry is unchanged.
    JAX inputs return host NumPy geometry; this interaction is outside JIT/autodiff.
    """
    return _open(geometry, read_only=False, **kwargs)
