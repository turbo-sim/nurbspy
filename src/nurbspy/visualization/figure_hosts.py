"""Desktop, headless PNG, and Dash hosts for the shared Figure renderer."""

import base64
from copy import deepcopy
import gzip
import json
from html import escape
import math
from pathlib import Path
import threading

from .desktop import _module_map, _script_json


def _packed_figure(figure):
    return base64.b64encode(gzip.compress(_script_json(
        dict(payload=figure._payload(), modules=_module_map())
    ).encode(), mtime=0)).decode()


def figure_html(figure, *, read_only=True, export=False, inline=True):
    """Use local data only; native windows fetch startup data over their bridge.

    Keeping the native HTML small avoids NavigateToString's UTF-16 size limit.
    Headless export can inline the same data without that WebView2 restriction.
    """
    packed = _packed_figure(figure) if inline else None
    return """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>""" + escape(figure.title) + """</title>
<style>[hidden]{display:none!important}html,body{height:100%;margin:0;background:white;font:14px system-ui}
body{display:flex;flex-direction:column;box-sizing:border-box;padding:16px;gap:12px}
#figure{min-height:0;flex:1}footer{display:flex;align-items:center;justify-content:flex-end;gap:10px}
#error{flex:1;color:#a7342b}#error:empty{display:none}
footer button{font:inherit;padding:9px 18px;border:1px solid #cbd7de;border-radius:7px;background:white;color:#294956;cursor:pointer}
footer button:disabled{opacity:.45;cursor:wait}#apply{background:#0072bd;color:white;border-color:#0072bd}
</style></head><body><div id="figure"></div><footer><span id="error" role="alert"></span>
<button id="close" disabled>Close</button><button id="apply" disabled>Apply</button></footer>
<script type="module">
const packed=""" + _script_json(packed) + ";const readOnly=" + str(read_only).lower() + ";const exporting=" + str(export).lower() + ";" + """
const close=document.getElementById('close'),apply=document.getElementById('apply'),error=document.getElementById('error');
close.textContent=readOnly?'Close':'Cancel';apply.hidden=readOnly;
function fail(e){error.textContent=e.message||String(e);window.nurbspyError=error.textContent;}
async function mount(){
    const startup=packed===null?await window.pywebview.api.startup_data():packed;
    const bytes=Uint8Array.from(atob(startup),c=>c.charCodeAt(0));
    const config=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).json();
    const map=document.createElement('script');map.type='importmap';map.textContent=JSON.stringify(config.modules);document.head.append(map);
    const {mountFigure}=await import('nurbspy/figure');
    const options={readOnly,export:exporting};
    if(!exporting){
        const api=window.pywebview.api;
        options.inspect=async(name,packet)=>{const r=await api.inspect(name,packet);if(r.error)throw new Error(r.error);return r;};
        options.evaluate=async(name,packet)=>{
            const r=await api.evaluate(name,packet);if(r.error)throw new Error(r.error);
            const a=Uint8Array.from(atob(r.vertices),c=>c.charCodeAt(0));
            return {...r,positions:new Float32Array(a.buffer)};
        };
        options.scenes=async state=>{const r=await api.scenes(state);if(r.error)throw new Error(r.error);return r;};
        options.onState=state=>api.capture(state);
    }else{
        document.querySelector('footer').hidden=true;document.body.style.padding='0';document.body.style.gap='0';
    }
    window.nurbspyFigure=await mountFigure('figure',config.payload,options);
    close.disabled=apply.disabled=false;
    return true;
}
function start(){window.nurbspyReady=mount().catch(e=>{fail(e);throw e;});}
if(exporting||window.pywebview?.api)start();else window.addEventListener('pywebviewready',start,{once:true});
async function finish(accept){
    try{
        close.disabled=apply.disabled=true;
        if(accept||readOnly){await window.nurbspyFigure.ready();const r=await window.pywebview.api.finish(window.nurbspyFigure.snapshot(),true);if(r.error)throw new Error(r.error);}
        else await window.pywebview.api.finish(null,false);
    }catch(e){close.disabled=apply.disabled=false;fail(e);}
}
close.addEventListener('click',()=>finish(false));apply.addEventListener('click',()=>finish(true));
window.addEventListener('keydown',e=>{if(e.key==='Escape'&&!exporting){e.preventDefault();finish(false);}});
</script></body></html>"""


class _FigureBridge:
    def __init__(self, figure, read_only):
        self._figure, self._read_only = figure, read_only
        self._models = figure._models()
        self._latest = None
        self._window = None
        self._accepted = False
        self._closed = False
        self._close_pending = False
        self._startup = None

    def startup_data(self):
        if self._startup is None:
            self._startup = _packed_figure(self._figure)
        return self._startup

    def evaluate(self, name, packet):
        try:
            if self._read_only or self._closed:
                raise ValueError("Figure is not accepting edits")
            positions, _, quality, revision = self._models[name]._render_update(packet)
            return dict(vertices=base64.b64encode(positions.tobytes()).decode(), quality=quality, revision=revision)
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            return dict(error=str(error))

    def inspect(self, name, packet):
        try:
            return self._models[name].inspect(packet)
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            return dict(error=str(error))

    def scenes(self, state):
        from .figure import Figure
        try:
            current = Figure.from_state(state)
            return current._payload()
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            return dict(error=str(error))

    def capture(self, state):
        if not self._closed:
            self._latest = deepcopy(state)

    def validated_state(self, state):
        """Validate against the session's fixed geometry types and topology."""
        from .figure import Figure
        current = Figure.from_state(state)
        if set(current.names) != set(self._figure.names):
            raise ValueError("Interactive sessions cannot add or remove objects")
        candidate_models = current._models()
        for name, model in self._models.items():
            candidate = candidate_models[name].initial_state
            model._validated_arrays(candidate)
            if self._read_only and candidate != model.initial_state:
                raise ValueError("A viewer cannot change geometry")
        normalized = current.to_state()
        normalized["objects"] = {name: normalized["objects"][name] for name in self._figure.names}
        return normalized

    def finish(self, state, accept):
        try:
            if self._closed:
                raise ValueError("Figure window has closed")
            if accept:
                self._figure._accept(self.validated_state(state))
                self._accepted = True
            self._closed = True
            self._window.destroy()
            return dict(ok=True)
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            return dict(error=str(error))

    def closing(self):
        # WinForms invokes this event synchronously on its GUI thread. Cancel
        # this first close, let JS finish capture asynchronously, then destroy
        # the window through finish(). Calling evaluate_js here would deadlock.
        if self._closed:
            return
        if not self._read_only:
            self._closed = True
            return
        if not self._close_pending:
            self._close_pending = True
            threading.Thread(target=self._request_view_close, daemon=True).start()
        return False

    def _request_view_close(self):
        try:
            self._window.evaluate_js('''(async()=>{
                try{
                    if(!window.nurbspyFigure){await window.pywebview.api.finish(null,false);return;}
                    await window.nurbspyFigure.ready();
                    const result=await window.pywebview.api.finish(window.nurbspyFigure.snapshot(),true);
                    if(result.error)throw new Error(result.error);
                }catch(error){
                    document.getElementById('error').textContent=error.message||String(error);
                    await window.pywebview.api.close_failed();
                }
            })()''')
        except Exception:
            self._close_pending = False

    def close_failed(self):
        self._close_pending = False


def open_figure(figure, *, read_only, width, height):
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Desktop visualization must run on Python's main thread")
    if any(type(n) is not int or n < 400 for n in (width, height)):
        raise ValueError("Window dimensions must be integers of at least 400 pixels")
    try:
        import webview
    except ImportError as error:
        raise ImportError("pywebview is required for desktop visualization; run poetry install or pip install nurbspy") from error
    bridge = _FigureBridge(figure, read_only)
    window = webview.create_window(figure.title, html=figure_html(figure, read_only=read_only, inline=False),
                                  js_api=bridge, width=width, height=height, min_size=(400, 400))
    bridge._window = window
    window.events.closing += bridge.closing
    webview.start(http_server=False, private_mode=True)
    if bridge._accepted and not read_only:
        geometries = {name: figure.geometry(name) for name in figure.names}
        return next(iter(geometries.values())) if len(geometries) == 1 else geometries
    return None


def _launch_browser(playwright):
    """Use an isolated headless engine, never a user's browser profile."""
    failures = []
    for channel in (None, 'msedge', 'chrome'):
        try:
            return playwright.chromium.launch(headless=True, channel=channel, timeout=15000,
                args=['--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        except Exception as error:
            failures.append(str(error).splitlines()[0])
    raise RuntimeError("PNG export needs a Chromium engine. Run `python -m playwright install chromium` "
                       "or install Microsoft Edge/Google Chrome. " + '; '.join(failures))


def render_png(figure, width, height, *, timeout):
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    from playwright.sync_api import sync_playwright
    logical = figure.rendering
    scale = min(width/logical['width'], height/logical['height'])
    css_width, css_height = width/scale, height/scale
    with sync_playwright() as playwright:
        browser = _launch_browser(playwright)
        try:
            page = browser.new_page(viewport=dict(width=math.ceil(css_width), height=math.ceil(css_height)),
                                    device_scale_factor=scale)
            page.set_default_timeout(timeout*1000)
            page.set_content(figure_html(figure, export=True), wait_until='load')
            page.wait_for_function('window.nurbspyReady !== undefined || window.nurbspyError')
            page.evaluate('''async milliseconds=>{
                let timer;
                try{await Promise.race([
                    (async()=>{await window.nurbspyReady;await document.fonts.ready;await window.nurbspyFigure.ready();})(),
                    new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error('Figure export timed out')),milliseconds);})
                ]);}finally{clearTimeout(timer);}
            }''', timeout*1000)
            state = page.evaluate('window.nurbspyFigure.snapshot()')
            data = page.screenshot(type='png', clip=dict(x=0, y=0, width=css_width, height=css_height),
                                   animations='disabled')
            return data, state
        finally:
            browser.close()


def render_frames(frames, directory, width, height, *, prefix, progress, timeout):
    """Stream native-evaluated scenes into one isolated renderer, then save PNGs."""
    from playwright.sync_api import sync_playwright
    from .figure import Figure
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    iterator = iter(frames)
    first = next(iterator, None)
    if first is None:
        return []
    if not isinstance(first, Figure):
        raise TypeError("Frames must be Figure objects")
    logical = dict(first.rendering)
    scale = min(width/logical['width'], height/logical['height'])
    css_width, css_height = width/scale, height/scale
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    with sync_playwright() as playwright:
        browser = _launch_browser(playwright)
        try:
            page = browser.new_page(viewport=dict(width=math.ceil(css_width), height=math.ceil(css_height)),
                                    device_scale_factor=scale)
            page.set_default_timeout(timeout*1000)
            page.set_content(figure_html(first, export=True), wait_until='load')
            page.wait_for_function('window.nurbspyReady !== undefined || window.nurbspyError')
            page.evaluate('async()=>{await window.nurbspyReady;await document.fonts.ready;}')
            current = first
            previous = None
            while current is not None:
                if not isinstance(current, Figure):
                    raise TypeError("Frames must be Figure objects")
                if any(current.rendering[key] != logical[key] for key in ('width', 'height')):
                    raise ValueError("All frames must have the same logical figure dimensions")
                if paths:
                    state = current.to_state()
                    resampled = any(state[key] != previous[key] for key in ('resolution', 'preview_resolution'))
                    changed = [name for name, item in state['objects'].items()
                               if resampled or name not in previous['objects']
                               or any(item[key] != previous['objects'][name][key] for key in ('geometry', 'display'))]
                    payload = current._payload(changed) if changed else dict(state=state, scenes={})
                    payload['patch'] = True
                    # One JSON string avoids Playwright's recursive per-number
                    # protocol encoding for large sampled scenes.
                    encoded = json.dumps(dict(payload=payload, milliseconds=timeout*1000),
                                         allow_nan=False, separators=(',', ':'))
                    page.evaluate('''async encoded=>{
                        const {payload,milliseconds}=JSON.parse(encoded);
                        let timer;
                        try{await Promise.race([
                            window.nurbspyFigure.update(payload),
                            new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error('Frame export timed out')),milliseconds);})
                        ]);}finally{clearTimeout(timer);}
                    }''', encoded)
                path = directory / f'{prefix}_{len(paths):05d}.png'
                page.screenshot(path=str(path), type='png',
                                clip=dict(x=0, y=0, width=css_width, height=css_height), animations='disabled')
                if current.camera is None:
                    current.set_camera(**page.evaluate('window.nurbspyFigure.snapshot().camera'))
                previous = current.to_state()
                paths.append(path)
                if progress is not None:
                    progress(len(paths), path)
                current = next(iterator, None)
            return paths
        finally:
            browser.close()


def figure_layout(component):
    from dash import dcc, html
    return html.Div([
        html.Div(id=component.viewport_id, style=dict(height=component.height, minWidth=0)),
        dcc.Store(id=component.scene_id, data=component._figure._payload()),
        dcc.Store(id=component.state_id, data=component._figure.to_state()),
        dcc.Store(id=component.ready_id),
    ], id=component.id, style=dict(width='100%', minWidth=0))


def register_figure(component, app):
    """Stateless host routes; each browser retains its own complete figure Store."""
    import json
    from pathlib import Path
    from dash import Input, Output
    from flask import Response, jsonify, request, send_from_directory
    from .figure import Figure
    registered = getattr(app, '_nurbspy_explorer_ids', set())
    if component.id in registered:
        raise ValueError(f'An explorer with id {component.id!r} is already registered')
    prefix = app.config.routes_pathname_prefix.rstrip('/')
    if not registered:
        assets = Path(__file__).parents[1] / 'web_assets'
        app.server.add_url_rule(prefix+'/_nurbspy_web/<path:filename>',
            endpoint=f'nurbspy_web_assets_{id(app)}',
            view_func=lambda filename: send_from_directory(assets, filename))
    base = f'/_nurbspy_web/figure/{component.id}'
    models = component._figure._models()

    def unpack():
        packet = request.get_json(silent=True)
        if not isinstance(packet, dict) or set(packet) != {'name', 'packet'} or not isinstance(packet['name'], str):
            raise ValueError('Expected object name and packet')
        return models[packet['name']], packet['packet']

    def inspect():
        try:
            model, packet = unpack()
            return jsonify(model.inspect(packet))
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            return jsonify(error=str(error)), 422

    def scenes():
        try:
            return jsonify(Figure.from_state(request.get_json(silent=True))._payload())
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            return jsonify(error=str(error)), 422

    def evaluate():
        try:
            model, packet = unpack()
            positions, _, quality, revision = model._render_update(packet)
            return Response(positions.tobytes(), mimetype='application/octet-stream', headers={
                'X-Nurbspy-Revision':str(revision), 'X-Nurbspy-Quality':quality, 'Cache-Control':'no-store'})
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            return jsonify(error=str(error)), 422

    routes = dict(inspect=inspect, scenes=scenes)
    if not component.read_only:
        routes['edit'] = evaluate
    for name, handler in routes.items():
        app.server.add_url_rule(prefix+base+'/'+name,
            endpoint=f'nurbspy_figure_{component.id}_{name}_{id(app)}', view_func=handler, methods=['POST'])
    options = dict(readOnly=component.read_only, stateStore=component.state_id,
                   base=app.get_relative_path(base))
    module = app.get_relative_path('/_nurbspy_web/figure.js')
    callback = """async function(payload){
        const {mountFigure}=await import(MODULE);
        const options=OPTIONS;
        const post=async(path,body)=>{
            const r=await fetch(options.base+'/'+path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
            if(!r.ok){const e=await r.json();throw new Error(e.error||'Figure request failed');}return r;
        };
        options.inspect=async(name,packet)=>(await post('inspect',{name,packet})).json();
        options.scenes=async state=>(await post('scenes',state)).json();
        if(!options.readOnly)options.evaluate=async(name,packet)=>{
            const r=await post('edit',{name,packet});return {revision:Number(r.headers.get('X-Nurbspy-Revision')),
                quality:r.headers.get('X-Nurbspy-Quality'),positions:new Float32Array(await r.arrayBuffer())};
        };
        options.onState=state=>window.dash_clientside.set_props(options.stateStore,{data:state});
        await mountFigure(ROOT,payload,options);return {mounted:true};
    }""".replace('MODULE', json.dumps(module)).replace('OPTIONS', json.dumps(options)).replace('ROOT', json.dumps(component.viewport_id))
    app.clientside_callback(callback, Output(component.ready_id, 'data'), Input(component.scene_id, 'data'))
    app._nurbspy_explorer_ids = registered | {component.id}
    return component
