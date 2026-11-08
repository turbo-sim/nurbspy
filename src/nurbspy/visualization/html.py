"""Self-contained, read-only Figure exports with precomputed variant switching."""

import base64
import gzip
import hashlib
from html import escape
from pathlib import Path
from .desktop import _module_map, _script_json


def export_html(figure, destination=None, *, variants=None):
    figures = {"Initial": figure} if variants is None else dict(variants)
    if not figures or any(not isinstance(name, str) or not name for name in figures):
        raise ValueError(
            "variants must be a nonempty mapping with nonempty string labels"
        )
    payloads = {}
    for name, current in figures.items():
        if not isinstance(current, type(figure)):
            raise TypeError("Every variant must be a Figure")
        print(f"Preparing HTML scene: {name}", flush=True)
        payloads[name] = current._payload(quality_levels=True)
    # Intern repeated arrays and scene metadata before compression. This keeps
    # long optimization histories from duplicating reference meshes/point clouds.
    pool = []
    known = {}

    def intern(value):
        if isinstance(value, dict):
            node = {key: intern(item) for key, item in value.items()}
        elif isinstance(value, list):
            node = [intern(item) for item in value]
        else:
            return value
        encoded = _script_json(node)
        if len(encoded) < 256:
            return node
        digest = hashlib.sha256(encoded.encode()).digest()
        if digest not in known:
            known[digest] = len(pool)
            pool.append(node)
        return {"__nrb_ref": known[digest]}

    tree = intern(dict(modules=_module_map(), payloads=payloads))
    packed = base64.b64encode(
        gzip.compress(_script_json(dict(tree=tree, pool=pool)).encode(), mtime=0)
    ).decode()
    document = (
        """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>"""
        + escape(figure.title)
        + """</title>
<style>html,body{height:100%;margin:0;font:14px system-ui;background:#fff}body{display:flex;flex-direction:column}
#toolbar{display:flex;gap:12px;align-items:center;padding:8px 12px;border-bottom:1px solid #dde5ec}#figure{flex:1;min-height:0}
select{font:inherit;padding:5px}#error{color:#a32f2f}.nrb-figure-export{border:0!important}</style></head>
<body><div id="toolbar"><label>View <select id="variant" aria-label="Figure variant"></select></label>
<label><input id="markers" type="checkbox" checked>Measurement points</label><span id="error" role="alert"></span></div><div id="figure"></div>
<script type="module">
const packed="""
        + _script_json(packed)
        + """;
const archive=await new Response(new Blob([Uint8Array.from(atob(packed),c=>c.charCodeAt(0))]).stream().pipeThrough(new DecompressionStream('gzip'))).json();
const cache=new Map();function unpack(value){
    if(value&&typeof value==='object'&&!Array.isArray(value)&&Object.keys(value).length===1&&Number.isInteger(value.__nrb_ref)){
        const id=value.__nrb_ref;if(!cache.has(id))cache.set(id,unpack(archive.pool[id]));return cache.get(id);
    }
    if(Array.isArray(value))return value.map(unpack);
    if(value&&typeof value==='object')return Object.fromEntries(Object.entries(value).map(([k,v])=>[k,unpack(v)]));
    return value;
}const config=unpack(archive.tree);
const map=document.createElement('script');map.type='importmap';map.textContent=JSON.stringify(config.modules);document.head.append(map);
const {mountFigure}=await import('nurbspy/figure');
const selector=document.querySelector('#variant'),names=Object.keys(config.payloads);
for(const name of names)selector.add(new Option(name,name));
let current=config.payloads[names[0]],queue=Promise.resolve();
window.nurbspyReady=mountFigure('figure',current,{readOnly:true,offline:true});
const viewer=await window.nurbspyReady;window.nurbspyFigure=viewer;
window.nurbspySelectVariant=name=>{
    if(!config.payloads[name])return Promise.reject(new Error('Unknown variant'));
    queue=queue.then(async()=>{
        await viewer.ready();const previous=viewer.snapshot(),next=structuredClone(config.payloads[name]);
        next.state.camera=previous.camera;
        for(const [key,item] of Object.entries(next.state.objects))if(previous.objects[key]){
            item.display=previous.objects[key].display;
            item.visible=previous.objects[key].visible;
        }
        await viewer.update(next);current=next;selector.value=name;
    });return queue;
};
window.nurbspySetLayers=choices=>{
    queue=queue.then(async()=>{
        await viewer.ready();const state=viewer.snapshot();
        for(const [name,settings] of Object.entries(choices.objects||{})){
            if(!state.objects[name])continue;
            if('visible' in settings)state.objects[name].visible=settings.visible;
            for(const key of ['control_points','control_polygon','opacity','grid'])if(key in settings)state.objects[name].display[key]=settings[key];
        }
        await viewer.update({state,scenes:current.scenes});current={state,scenes:current.scenes};
        if('markers' in choices)viewer.setMarkersVisible(choices.markers);
    });return queue;
};
selector.onchange=()=>window.nurbspySelectVariant(selector.value).catch(e=>document.querySelector('#error').textContent=e.message);
document.querySelector('#markers').onchange=e=>viewer.setMarkersVisible(e.target.checked);
window.nurbspyReportReady=true;
</script></body></html>"""
    )
    if destination is None:
        return document
    path = Path(destination)
    path.write_text(document, encoding="utf-8")
    return path
