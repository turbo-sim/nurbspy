/* Embeddable NURBS editing viewport. Geometry evaluation stays in Python. */
import * as THREE from './vendor/three.module.js';
import {OrbitControls} from './vendor/OrbitControls.js';
import {TrackballControls} from './vendor/TrackballControls.js';
import {ArcballControls} from './vendor/ArcballControls.js';
import {createViewCube} from './viewcube.js';
import {ScientificDisplay} from './science.js';
import {EditingControls} from './editing.js';
import {createInspector,syncInspector} from './panels.js';
import {lineGeometry,updateLinePositions,wideLine,boundaryIndices,referenceLayers,planarFrame,createPlanarAxes,projectionPlanes,planarHandleRadius,spatialHandleRadius,renderLayeredScene} from './display.js';

// Turbodash plotting colours: MATLAB blue and the clear orange in COLORS_PYTHON.
// Local constants avoid introducing a dependency on the turbodash application.
const colors = Object.freeze({curve:'#000000',control:'#ff7f0e',selected:'#2ca02c',surface:'#0072BD'});
const drawOrder = Object.freeze({geometry:0,controlPolygon:10,handle:20,outline:21});
const viewers = new Map();
const clone = value => JSON.parse(JSON.stringify(value));

// Transport adapters return display vertices, never evaluate NURBS in JavaScript.
export function httpEvaluator(endpoint) {
    return async (packet,signal) => {
        const response=await fetch(endpoint,{
            method:'POST',headers:{'Content-Type':'application/json'},signal,
            body:JSON.stringify(packet),
        });
        if (!response.ok) {
            const error=await response.json(); throw new Error(error.error || 'Python rejected the edit.');
        }
        return {revision:Number(response.headers.get('X-Nurbspy-Revision')),
            quality:response.headers.get('X-Nurbspy-Quality'),
            positions:new Float32Array(await response.arrayBuffer())};
    };
}
const css = `
:root{--nrb-accent:#0072bd;--nrb-selected:#2ca02c}
.nrb-web{box-sizing:border-box;display:flex;flex-direction:column;position:relative;overflow:hidden;min-width:0;min-height:320px;border:1px solid #dbe3e8;border-radius:12px;background:#fff;color:#203440;font:13px/1.4 system-ui,sans-serif}
.nrb-web *{box-sizing:border-box}.nrb-web [hidden]{display:none!important}
.nrb-web .nrb-heading{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;padding:13px 17px;border-bottom:1px solid #e7edf1}.nrb-web h3{font-size:16px;font-weight:650;margin:0;min-width:0;overflow-wrap:anywhere}
.nrb-web button,.nrb-web select,.nrb-web input[type=number]{font:inherit;color:inherit;background:#fff;border:1px solid #ccd7de;border-radius:6px;padding:5px 8px;min-height:30px}
.nrb-web button{cursor:pointer}.nrb-web button:hover{background:#edf5fb}
.nrb-web button:focus-visible,.nrb-web select:focus-visible,.nrb-web input:focus-visible,.nrb-web summary:focus-visible{outline:2px solid var(--nrb-accent);outline-offset:2px}
.nrb-web button:disabled,.nrb-web select:disabled,.nrb-web input:disabled{opacity:.5;cursor:default}
.nrb-web label{display:flex;align-items:center;gap:6px}
.nrb-web input[type=range]{width:82px;accent-color:var(--nrb-accent)}.nrb-web input[type=number]{width:65px}
.nrb-web .nrb-camera-tools,.nrb-web .nrb-camera-actions,.nrb-web .nrb-panel-actions{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.nrb-web .nrb-camera-tools{margin-left:auto;max-width:100%;justify-content:flex-end;gap:10px}
.nrb-web .nrb-navigation{display:flex;align-items:center;gap:8px;white-space:nowrap}.nrb-web .nrb-navigation select{min-width:112px}
.nrb-web .nrb-panel-actions{border-left:1px solid #dbe3e8;padding-left:10px;gap:6px}
.nrb-web .nrb-control-label{font-size:11px;color:#6b7d88}
.nrb-web .nrb-panel-toggle{font-weight:550;white-space:nowrap}
.nrb-web .nrb-panel-toggle[aria-expanded=true]{background:#edf5fb;border-color:#9dbfd6;color:#0067a7}
.nrb-web .nrb-panel-toggle[aria-expanded=true]:hover{background:#e2eff8}
.nrb-web .nrb-workspace{display:flex;position:relative;flex:1;min-height:160px;min-width:0;--nrb-panel-width:min(310px,52%)}
.nrb-web .nrb-canvas{position:relative;flex:1;min-width:0;min-height:160px;overflow:hidden;background:#fff}
.nrb-web .nrb-renderer{position:absolute;inset:0;display:block;width:100%;height:100%;touch-action:none;outline:none;z-index:0}
.nrb-web .nrb-sidebar{flex:0 0 var(--nrb-panel-width);min-width:0;display:flex;flex-direction:column;border-left:1px solid #dde5eb;background:#fff;overflow:hidden}
.nrb-web .nrb-panel-heading{display:flex;align-items:center;justify-content:space-between;padding:13px 16px 10px;border-bottom:1px solid #e7edf1}
.nrb-web .nrb-panel-heading h4{font-size:14px;font-weight:650;margin:0}
.nrb-web .nrb-panel-heading button{font-size:20px;line-height:1;min-height:26px;padding:1px 6px;border:0;color:#6b7d88}
.nrb-web .nrb-config-panel{flex:1;overflow:auto;min-height:0;padding:0 16px 16px;scrollbar-width:thin;scrollbar-gutter:stable;container-type:inline-size;animation:nrb-panel-in .15s ease-out}
.nrb-web input.nrb-edit-weight-slider[type=range]{width:100%;margin:3px 0}
.nrb-web .nrb-config-group{border-bottom:1px solid #e6edf2}.nrb-web .nrb-config-group:last-of-type{border-bottom:0}
.nrb-web .nrb-config-group>summary{list-style:none;display:flex;align-items:center;justify-content:space-between;padding:13px 0;cursor:pointer;font-size:12px;font-weight:650;color:#314e61;user-select:none}
.nrb-web .nrb-config-group>summary::-webkit-details-marker{display:none}.nrb-web .nrb-config-group>summary:after{content:'›';font-size:17px;font-weight:400;color:#8294a0}.nrb-web .nrb-config-group[open]>summary:after{transform:rotate(90deg)}
.nrb-web .nrb-config-body{display:flex;flex-direction:column;gap:11px;padding:0 0 15px}
.nrb-web .nrb-science-row{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.15fr);align-items:center;gap:10px;min-height:30px;font-size:12px;overflow-wrap:anywhere}
.nrb-web .nrb-science-row>select,.nrb-web .nrb-science-row>input[type=number]{width:100%;min-width:0;max-width:none;font:inherit}
.nrb-web .nrb-science-row>input[type=number]{font-variant-numeric:tabular-nums}
.nrb-web .nrb-science-row>input[type=checkbox]{justify-self:start;margin:0}
.nrb-web .nrb-config-panel input[type=color]{width:34px;height:26px;padding:2px;border:1px solid #ccd7de;border-radius:5px;background:white;cursor:pointer;flex-shrink:0}
.nrb-web input[type=checkbox]{accent-color:var(--nrb-accent)}
.nrb-web .nrb-config-panel small,.nrb-web .nrb-science-note{font-size:11px;color:#71838e;line-height:1.5;white-space:normal}
.nrb-web .nrb-science-note{display:block;margin-top:12px}.nrb-web .nrb-colour-reset{margin-top:12px;width:100%;font-size:12px}
.nrb-web .nrb-colorbar{position:absolute;right:14px;top:17px;z-index:2;width:174px;background:#fffffff2;border:1px solid #e0e6eb;border-radius:6px;padding:9px;pointer-events:none;font:11px/1.4 system-ui,sans-serif;color:#36434c}
.nrb-web .nrb-colorbar-gradient{height:10px;margin:5px 0}.nrb-web .nrb-colorbar-range{display:flex;justify-content:space-between;font-variant-numeric:tabular-nums}
.nrb-web .nrb-frame-legend{position:absolute;right:14px;bottom:12px;z-index:2;background:#ffffffdb;padding:3px 7px;border-radius:4px;font-size:11px;pointer-events:none}
.nrb-web .nrb-planar-axes{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:1;display:none;overflow:visible}
.nrb-web .nrb-planar-axes path,.nrb-web .nrb-planar-axes rect{stroke:#4f5c66;stroke-width:1;fill:none}
.nrb-web .nrb-planar-axes text{font:11px system-ui,sans-serif;fill:#36434c;stroke:none;font-variant-numeric:tabular-nums}
.nrb-web .nrb-planar-axes .nrb-axis-title{font:italic 13px system-ui,sans-serif}
.nrb-web .nrb-footer{position:absolute;top:12px;left:12px;max-width:calc(100% - 24px);z-index:3;pointer-events:none;padding:5px 9px;background:#fffffff2;border:1px solid #e0e6eb;border-radius:5px;font-size:11px}
.nrb-web .nrb-status{color:#597483;text-align:right}.nrb-web .nrb-status.nrb-error{color:#b23b32}
.nrb-web .nrb-selection-tools{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:12px}
.nrb-web .nrb-selection-box{position:absolute;z-index:5;pointer-events:none;border:1px dashed #0072bd;background:#0072bd18}
.nrb-web.nrb-updating .nrb-status{color:#9c6a19}.nrb-web .nrb-failure{padding:25px;color:#9b342e}
@keyframes nrb-panel-in{from{opacity:0;transform:translateX(10px)}to{opacity:1;transform:translateX(0)}}
@media(prefers-reduced-motion:reduce){.nrb-web .nrb-config-panel{animation:none}}
@container(max-width:230px){.nrb-web .nrb-science-row:has(>select),.nrb-web .nrb-science-row:has(>input[type=number]){grid-template-columns:minmax(0,1fr);gap:5px}}
@media(max-width:650px){.nrb-web .nrb-heading{padding:10px;gap:8px}}

/* ViewCube styles adapted from turbo_moc; MIT attribution in viewcube.js. */
.nrb-web .orientation-widget {position:absolute;bottom:10px;left:10px;width:144px;z-index:2;
  border:1px solid #d6e1e6;border-radius:9px;background:#ffffffeb;box-shadow:0 2px 8px #173c5010;
  user-select:none;touch-action:none;color:#345563}
.nrb-web .orientation-widget[hidden] {display:none}
.nrb-web .orientation-widget svg {display:block;width:100%;height:auto;overflow:visible;border-radius:8px}
.nrb-web .orientation-widget svg:focus-visible {outline:2px solid #477db3;outline-offset:2px}
.nrb-web .orientation-face {cursor:pointer;outline:none}
.nrb-web .orientation-face-fill {stroke:#8298a2;stroke-width:.8;stroke-linejoin:round;fill:#eff4f6}
.nrb-web .orientation-face-fill[data-axis=X] {fill:#edf3f8}
.nrb-web .orientation-face-fill[data-axis=Y] {fill:#e4edef}
.nrb-web .orientation-face-fill[data-axis=Z] {fill:#fafcfd}
.nrb-web .orientation-face text {font:600 12px system-ui,sans-serif;fill:#345563;pointer-events:none}
.nrb-web .orientation-edge .orientation-face-fill {fill:#d9e3e8}
.nrb-web .orientation-corner .orientation-face-fill {fill:#c9d7df}
.nrb-web .orientation-face:hover polygon,.nrb-web .orientation-face:focus-visible polygon {fill:#c9e5ef;stroke:#376a7d;stroke-width:1.7}
.nrb-web .orientation-control {cursor:pointer;outline:none}
.nrb-web .orientation-control circle {fill:#f8fafb;stroke:#dce5e9;stroke-width:.8}
.nrb-web .orientation-control path {fill:none;stroke:#647f8d;stroke-width:1.65;stroke-linecap:round;stroke-linejoin:round;pointer-events:none}
.nrb-web .orientation-control:hover circle,.nrb-web .orientation-control:focus-visible circle {fill:#e1eff5;stroke:#648d9f}
.nrb-web .orientation-control:hover path,.nrb-web .orientation-control:focus-visible path {stroke:#244f65}
.nrb-web .orientation-axis {stroke-width:1.8;stroke-linecap:round}
.nrb-web .orientation-axis[data-axis=X] {stroke:#b65353}
.nrb-web .orientation-axis[data-axis=Y] {stroke:#3b8765}
.nrb-web .orientation-axis[data-axis=Z] {stroke:#477db3}
.nrb-web .orientation-axis-label {font:600 10px system-ui,sans-serif;text-anchor:middle}
.nrb-web .orientation-axis-label[data-axis=X] {fill:#b65353}
.nrb-web .orientation-axis-label[data-axis=Y] {fill:#3b8765}
.nrb-web .orientation-axis-label[data-axis=Z] {fill:#477db3}
.nrb-web .orientation-footer {display:flex;align-items:center;justify-content:flex-end;padding:0 9px 8px;gap:6px}
.nrb-web .orientation-footer button {font-size:9px;font-weight:600;letter-spacing:.3px;padding:3px 6px;border-radius:4px;min-width:0}
.nrb-web .orientation-footer button:focus-visible {outline:2px solid #477db3;outline-offset:2px}

`;
function element(tag, attributes = {}, text) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attributes)) {
        if (key === 'class') node.className = value;
        else node.setAttribute(key, value);
    }
    if (text !== undefined) node.textContent = text;
    return node;
}

function modelPoints(model, shape) {
    const count = shape.reduce((a, b) => a * b, 1);
    return Array.from({length: count}, (_, index) => [0, 1, 2].map(d => {
        if (d >= model.control_points.length) return 0;
        return shape.length === 1 ? model.control_points[d][index] :
            model.control_points[d][Math.floor(index / shape[1])][index % shape[1]];
    }));
}

export class Explorer {
    constructor(root, initial, options = {}) {
        this.root = root;
        this.options = options;
        this.readOnly = Boolean(options.readOnly);
        this.dimension = 3;
        this.viewMode = '3d';
        this.renderFrame = null;
        this.geometryCache = new Map();
        this.topologies = initial.topologies || {full: {triangles: initial.triangles, count: initial.points.length}};
        this.acceptedModel = clone(initial.model);
        this.lastRevision = -1;
        this.nextRevision = 0;
        this.localGeneration = 0;
        this.pending = null;
        this.inFlight = null;
        this.sendTimer = null;
        this.selected = 0;
        this.selectedIndices = new Set([0]);
        this.handles = [];
        this.drag = null;
        this.original = clone(initial.model);
        this.buildDOM(initial);
        this.buildScene(initial);
        this.apply(initial);
        if(!options.shared)this.setViewMode(initial.display?.view || '3d');
        this.removalObserver = new MutationObserver(() => {
            if (!root.isConnected) this.dispose();
        });
        this.removalObserver.observe(document.body, {childList: true, subtree: true});
    }

    buildDOM(initial) {
        if(!document.getElementById('nrb-web-style')){
            const style=element('style',{id:'nrb-web-style'});style.textContent=css;document.head.appendChild(style);
        }
        this.root.classList.add('nrb-web');
        const heading=element('div',{class:'nrb-heading'});
        heading.append(element('h3',{},initial.title||'NURBS geometry'));
        const cameraTools=element('div',{class:'nrb-camera-tools'});
        const cameraActions=element('div',{class:'nrb-camera-actions'});
        this.navigationLabel=element('label',{class:'nrb-navigation'});
        this.navigationLabel.append(element('span',{class:'nrb-control-label'},'Navigation'));
        this.navigationSelect=element('select',{'aria-label':'Navigation mode',title:'Choose how the camera rotates'});
        const choices=[['arcball','Arcball','Free rotation around a virtual sphere'],['orbit','Orbit','Turntable rotation around the current up direction'],['trackball','Trackball','Free camera-relative tumbling']];
        for(const [value,label,description] of choices){
            this.navigationSelect.append(element('option',{value,title:description},label));
        }
        this.navigationSelect.addEventListener('change',()=>this.setNavigation(this.navigationSelect.value));
        this.navigationLabel.append(this.navigationSelect);
        const fit=element('button',{type:'button',title:'Fit all control points in the view'},'Fit');
        cameraActions.append(this.navigationLabel,fit);
        if(!this.readOnly){
            this.undoButton=element('button',{type:'button',disabled:'','aria-label':'Undo geometry edit',title:'Undo (Ctrl/Command+Z)'},'Undo');
            this.redoButton=element('button',{type:'button',disabled:'','aria-label':'Redo geometry edit',title:'Redo (Ctrl/Command+Shift+Z)'},'Redo');
            this.resetButton=element('button',{type:'button',title:'Restore the original control points and weights'},'Reset');
            cameraActions.append(this.undoButton,this.redoButton,this.resetButton);
        }
        const panelActions=element('div',{class:'nrb-panel-actions','aria-label':'Inspector panels'});
        cameraTools.append(cameraActions,panelActions);
        heading.append(cameraTools);

        const workspace=element('div',{class:'nrb-workspace'});
        this.viewport=element('div',{class:'nrb-canvas'});
        this.panels={display:createInspector(this,'display','Display','display settings')};
        this.sidebar=this.panels.display.sidebar;this.configPanel=this.panels.display.body;
        this.displayToggle=this.panels.display.toggle;
        this.configSections={};
        for(const [key,title] of [['view','View'],['geometry','Geometry'],['controls','Control points'],['scientific','Scientific colouring'],['extras',initial.triangles?.length?'Surface mesh':'Frenet–Serret frames']]){
            const group=element('details',{class:'nrb-config-group',open:''});group.append(element('summary',{},title));
            const body=element('div',{class:'nrb-config-body'});group.append(body);this.configPanel.append(group);this.configSections[key]=body;
        }
        const row=(section,label,input)=>{const field=element('label',{class:'nrb-science-row'},label);field.append(input);section.append(field);return input;};
        const toggle=(section,label,checked)=>{
            const input=element('input',{type:'checkbox','aria-label':label});input.checked=checked;return row(section,label,input);
        };
        this.viewModeSelect=element('select',{'aria-label':'Projection'});
        for(const [value,label] of [['3d','3D perspective'],['xy','X–Y'],['yz','Y–Z'],['zx','Z–X']])this.viewModeSelect.append(element('option',{value},label));
        row(this.configSections.view,'Projection',this.viewModeSelect);
        this.axesCheck=toggle(this.configSections.view,'Axes with ticks',false);
        this.gridCheck=toggle(this.configSections.view,'Reference grid',true);
        this.configSections.view.append(element('small',{},'Ticks are available in the X–Y, Y–Z, and Z–X projections.'));
        if(initial.triangles?.length)this.edgeCheck=toggle(this.configSections.geometry,'Boundary curves',true);
        this.netCheck=toggle(this.configSections.controls,'Control polygon',true);
        this.handleCheck=toggle(this.configSections.controls,'Control points',true);
        workspace.append(this.viewport,this.sidebar);panelActions.append(this.displayToggle);
        if(!this.readOnly){
            this.editing=new EditingControls(this);this.panels.edit=this.editing.panel;
            workspace.append(this.editing.panel.sidebar);panelActions.append(this.editing.panel.toggle);
        }
        this.planarAxes=createPlanarAxes(this.viewport);
        this.scientific=new ScientificDisplay(this,this.configSections,initial);
        this.statusBar=element('div',{class:'nrb-footer',hidden:''});
        this.status=element('span',{class:'nrb-status',role:'status'});
        // Edit feedback must not resize the canvas or shift the drag coordinates.
        this.statusBar.append(this.status);this.viewport.append(this.statusBar);
        this.root.replaceChildren(heading,workspace);
        this.onInspectorKey=event=>{
            if(this.editing?.keyDown(event))return;
            if(event.key!=='Escape'||!this.activePanel)return;
            event.preventDefault();event.stopPropagation();
            const panel=this.panels[this.activePanel];this.setPanelOpen(false,panel.kind);panel.toggle.focus();
        };
        this.root.addEventListener('keydown',this.onInspectorKey);
        this.viewModeSelect.addEventListener('change',()=>this.setViewMode(this.viewModeSelect.value));
        this.gridCheck.addEventListener('change',()=>{this.syncDisplay();this.render();});
        this.axesCheck.addEventListener('change',()=>{this.syncDisplay();this.resize();});
        this.edgeCheck?.addEventListener('change',()=>{this.surfaceEdges?.forEach(edge=>edge.visible=this.edgeCheck.checked);this.render();});
        this.netCheck.addEventListener('change',()=>{if(this.net)this.net.visible=this.netCheck.checked;this.render();});
        this.handleCheck.addEventListener('change',()=>{this.handles.forEach(handle=>handle.visible=this.handleCheck.checked);this.render();});
        fit.addEventListener('click',()=>this.fit());
        this.resetButton?.addEventListener('click',()=>this.editing.reset());
        this.undoButton?.addEventListener('click',()=>this.editing.travel());
        this.redoButton?.addEventListener('click',()=>this.editing.travel(true));
    }

    setPanelOpen(open,kind='display') {
        if(!this.panels[kind])return;
        this.activePanel=open?kind:null;
        for(const panel of Object.values(this.panels))syncInspector(panel,this.activePanel);
        this.resize();
    }

    buildScene(initial) {
        if(this.options?.shared){
            const shared=this.options.shared;
            this.scene=new THREE.Group();this.camera=shared.camera;this.controls=shared.controls;
            this.renderer=shared.renderer;this.is2D=shared.is2D;this.viewMode=shared.viewMode;
            this.render=()=>shared.render();this.resize=()=>{};this.fit=()=>{};
            this.setHandleStyle();return;
        }
        this.scene = new THREE.Scene();
        this.referenceScene = new THREE.Scene();
        this.dimension=3;
        this.is2D = false;
        this.viewMode = '3d';
        this.camera = this.is2D ? new THREE.OrthographicCamera(-3, 3, 3, -3, .001, 10000) :
            new THREE.PerspectiveCamera(38, 1, .001, 10000);
        // OrbitControls caches this axis at construction, so configure it first.
        this.camera.up.set(0,this.is2D ? 1 : 0,this.is2D ? 0 : 1);
        this.camera.position.set(3,-4,3); this.camera.lookAt(0,0,0);
        this.renderer = new THREE.WebGLRenderer({antialias: true, alpha: false});
        this.renderer.autoClear=false;
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        this.renderer.setClearColor?.('#ffffff',1);
        this.renderer.domElement.setAttribute('aria-label', 'Interactive NURBS geometry viewport');
        this.renderer.domElement.setAttribute('tabindex', '0');
        this.renderer.domElement.classList.add('nrb-renderer');
        this.viewport.prepend(this.renderer.domElement);
        this.setNavigation(this.is2D ? 'orbit' : 'arcball');
        if (!this.is2D) this.ensureViewCube();
        this.scene.add(new THREE.HemisphereLight(0xffffff, 0x6b8798, 2.5));
        const directional = new THREE.DirectionalLight(0xffffff, 2.2);
        directional.position.set(2, -3, 7); this.scene.add(directional);
        if(this.gridCheck) {
            this.references=referenceLayers(this.is2D);
            this.referenceScene.add(this.references.floor,this.references.tickGrid);
        }
        // Handles must share the transparent draw pass with the polygon/surface;
        // renderOrder then guarantees their opaque white faces cover those lines.
        this.setHandleStyle();
        this.syncDisplay();
        this.raycaster = new THREE.Raycaster();
        this.pointer = new THREE.Vector2();
        this.onDown = event => this.pointerDown(event);
        this.onMove = event => this.pointerMove(event);
        this.onUp = event => this.pointerUp(event);
        this.onNavigationInput = () => this.render();
        this.renderer.domElement.addEventListener('pointerdown', this.onDown, true);
        this.renderer.domElement.addEventListener('pointermove', this.onMove, true);
        this.renderer.domElement.addEventListener('pointerup', this.onUp, true);
        this.renderer.domElement.addEventListener('pointercancel', this.onUp, true);
        this.renderer.domElement.addEventListener('lostpointercapture', this.onUp, true);
        this.renderer.domElement.addEventListener('pointermove',this.onNavigationInput);
        this.renderer.domElement.addEventListener('wheel',this.onNavigationInput,{passive:true});
        this.resizeObserver = new ResizeObserver(() => this.resize());
        this.resizeObserver.observe(this.viewport);
    }

    setHandleStyle() {
        const geometries=new Set([this.pointGeometry,this.outlineGeometry].filter(Boolean));
        const materials=new Set([this.pointMaterial,this.selectedMaterial,this.outlineMaterial,this.selectedOutlineMaterial].filter(Boolean));
        this.pointGeometry = this.is2D ? new THREE.CircleGeometry(1,32) : new THREE.SphereGeometry(1,32,20);
        this.outlineGeometry=this.outlineMaterial=this.selectedOutlineMaterial=null;
        if (this.is2D) {
            this.pointMaterial = new THREE.MeshBasicMaterial({color:'#ffffff',transparent:true,opacity:1,depthTest:false,depthWrite:false});
            this.selectedMaterial = this.pointMaterial;
            this.outlineGeometry = new THREE.RingGeometry(.68,1.08,32);
            this.outlineMaterial = new THREE.MeshBasicMaterial({color:colors.control,transparent:true,opacity:1,depthTest:false,depthWrite:false});
            this.selectedOutlineMaterial = new THREE.MeshBasicMaterial({color:colors.selected,transparent:true,opacity:1,depthTest:false,depthWrite:false});
        } else {
            const finish = {roughness:.22,metalness:.18,clearcoat:.8,clearcoatRoughness:.12,transparent:true,opacity:1,depthTest:false,depthWrite:false};
            this.pointMaterial = new THREE.MeshPhysicalMaterial({color:colors.control,...finish});
            this.selectedMaterial = new THREE.MeshPhysicalMaterial({color:colors.selected,emissive:colors.selected,emissiveIntensity:.18,...finish});
        }
        for(const handle of this.handles||[]) this.styleHandle(handle);
        geometries.forEach(geometry=>geometry.dispose());materials.forEach(material=>material.dispose());
    }

    styleHandle(handle) {
        handle.geometry=this.pointGeometry;handle.material=this.pointMaterial;
        handle.quaternion.copy(this.is2D ? this.camera.quaternion : new THREE.Quaternion());
        if(handle.userData.outline){handle.remove(handle.userData.outline);delete handle.userData.outline;}
        if(this.is2D){
            const outline=new THREE.Mesh(this.outlineGeometry,this.outlineMaterial);
            outline.renderOrder=drawOrder.outline;handle.add(outline);handle.userData.outline=outline;
        }
    }

    ensureViewCube() {
        if(this.viewCube) return;
        this.viewCube=createViewCube(this.viewport,{
            interactionElement:this.renderer.domElement,
            readCamera:()=>({position:this.camera.position.toArray(),target:this.controls.target.toArray(),up:new THREE.Vector3(0,1,0).applyQuaternion(this.camera.quaternion).toArray()}),
            writeCamera:(frame,finished)=>this.writeView(frame,finished),
            onCancel:()=>{this.cameraAnimating=false;this.setNavigation(this.navigation,{cancelCube:false});},
        });
        this.viewCube.setHome([1.3,-1.7,1.5],[0,0,1]);
    }

    setViewMode(mode) {
        if(mode!=='3d' && !projectionPlanes[mode]) throw new Error('Unknown view mode.');
        const planar=mode!=='3d';
        if(mode===this.viewMode) return;
        this.viewCube?.cancel(false);if(this.drag) this.endDrag();
        if(!this.is2D) this.lastSpatialNavigation=this.navigation;
        this.is2D=planar;this.viewMode=mode;
        this.camera=planar ? new THREE.OrthographicCamera(-3,3,3,-3,.001,10000) : new THREE.PerspectiveCamera(38,1,.001,10000);
        this.camera.up.set(0,planar?1:0,planar?0:1);this.camera.position.set(3,-4,3);
        this.setNavigation(planar?'orbit':this.lastSpatialNavigation||'arcball',{cancelCube:false});
        if(!planar) this.ensureViewCube();
        this.setHandleStyle();this.syncDisplay();this.fit();
        this.scientific?.applyColors();
    }

    syncDisplay() {
        if(this.root.classList) this.root.classList.toggle('nrb-3d',!this.is2D);
        if(this.viewModeSelect) this.viewModeSelect.value=this.viewMode;
        if(this.navigationLabel) this.navigationLabel.hidden=this.is2D;
        if(this.axesCheck) this.axesCheck.disabled=!this.is2D;
        this.editing?.syncProjection();
        if(this.references){
            this.references.floor.quaternion.setFromUnitVectors(new THREE.Vector3(0,0,1),new THREE.Vector3(...(projectionPlanes[this.viewMode]?.direction || [0,0,1])));
        }
        if(this.references) this.references.floor.visible=Boolean(this.gridCheck?.checked && !(this.is2D&&this.axesCheck?.checked));
        this.surfaceEdges?.forEach(edge=>{edge.material.depthTest=!this.is2D;});
        if(this.viewCube?.host) this.viewCube.host.hidden=this.is2D;
    }

    plotRectangle(width=this.viewport.clientWidth,height=this.viewport.clientHeight) {
        const rectangle=this.is2D&&this.axesCheck?.checked ? planarFrame(width,height) : {left:0,top:0,bottom:0,width,height};
        const aspect=this.options?.aspect;
        if(aspect){
            const fittedWidth=Math.min(rectangle.width,rectangle.height*aspect),fittedHeight=fittedWidth/aspect;
            rectangle.left+=(rectangle.width-fittedWidth)/2;rectangle.top+=(rectangle.height-fittedHeight)/2;
            rectangle.bottom+=(rectangle.height-fittedHeight)/2;
            rectangle.width=fittedWidth;rectangle.height=fittedHeight;
        }
        return rectangle;
    }

    setNavigation(mode, {target, cancelCube=true} = {}) {
        if (this.is2D) mode = 'orbit';
        const Control = {orbit:OrbitControls,trackball:TrackballControls,arcball:ArcballControls}[mode];
        if (!Control) throw new Error('Unknown navigation mode.');
        if (cancelCube) this.viewCube?.cancel(false);
        this.cameraAnimating = false;
        if (this.drag) this.endDrag();
        target = target || this.controls?.target.clone() || new THREE.Vector3();
        const position=this.camera.position.clone(), up=this.camera.up.clone(), zoom=this.camera.zoom;
        this.controls?.dispose();
        this.controls = new Control(this.camera,this.renderer.domElement);
        this.camera.position.copy(position); this.camera.up.copy(up); this.camera.zoom=zoom;
        this.controls.target.copy(target);
        if (mode==='arcball') {
            this.controls.enableAnimations=false; this.controls.enableGizmos=false;this.controls.enableRotate=true;
            this.controls.setCamera(this.camera);
        } else if (mode==='trackball') {
            this.controls.staticMoving=true; this.controls.rotateSpeed=2;
            this.controls.handleResize();
        } else {
            this.controls.enableDamping=false; this.controls.enableRotate=!this.is2D;
            this.controls.screenSpacePanning=true;
        }
        this.navigation=mode;
        this.root.dataset.navigation=mode;
        if (this.navigationSelect) this.navigationSelect.value=mode;
        this.camera.updateProjectionMatrix(); this.controls.update();
        this.controls.addEventListener('change',()=>this.render());
        this.render();
    }

    writeView(frame, finished) {
        this.cameraAnimating=!finished;
        this.camera.position.fromArray(frame.position); this.camera.up.fromArray(frame.up);
        this.controls.target.fromArray(frame.target);
        this.camera.lookAt(this.controls.target);
        if (finished) this.setNavigation(this.navigation,{cancelCube:false});
        this.render();
    }

    setStatus(message, error = false) {
        this.status.textContent = message;
        this.status.classList.toggle('nrb-error', error);
        if(this.statusBar)this.statusBar.hidden=!message;
    }

    apply(data) {
        if (this.disposed || data.revision < this.lastRevision) return;
        const first = this.lastRevision === -1;
        const flight = this.inFlight;
        const acknowledged = flight && data.revision === flight.revision;
        const newerLocal = acknowledged && this.localGeneration > flight.generation;
        this.lastRevision = data.revision;
        this.nextRevision = Math.max(this.nextRevision, data.revision);
        this.editStore = data.edit_store || this.editStore;
        this.shape = data.control_shape || this.shape;
        this.root.dataset.evaluatedRevision = String(data.revision);
        if (!data.error) this.acceptedModel = clone(data.model);
        if (!this.inFlight || acknowledged) {
            if (!newerLocal && !this.pending) {
                this.model = clone(data.model);
                this.moveHandles(data.control_points || modelPoints(this.model, this.shape));
            }
            if (acknowledged) this.inFlight = null;
        }
        if (!data.error) this.updateGeometry(data);
        this.updateNet(data.control_lines);
        // The first net is created after geometry colouring; apply its configured
        // colour now too, so frame zero matches subsequent/restored frames.
        this.scientific?.applyColors();
        if(data.error&&this.editing)this.editing.normalKey=null;
        this.updateSelection();
        const working = this.pending || this.inFlight || this.drag;
        this.setStatus(data.error || (working ? 'Updating in Python…' : ''), Boolean(data.error));
        this.root.classList?.toggle('nrb-updating',Boolean(working));
        if (first) {this.resize(); this.fit();}
        this.render();
        // Expensive host analysis runs after a full accepted edit, not for every
        // transient preview. The browser performs no scientific evaluation.
        if (!data.error && acknowledged && flight.quality === 'full' && !newerLocal && !this.pending) {
            this.options.onChange?.(clone(data.model));
        }
        if(acknowledged)this.editing?.acknowledge(data,flight);
        if (this.pending && !this.inFlight) this.flush();
        this.editing?.updateHistory();
        this.options.onBusyChange?.(Boolean(this.pending || this.inFlight || this.drag));
    }

    moveHandles(points) {
        if (this.handles.length !== points.length) {
            this.handles.forEach(handle => this.scene.remove(handle));
            this.handles = points.map((point, index) => {
                const handle = new THREE.Mesh(this.pointGeometry, this.pointMaterial);
                handle.userData.index = index; handle.renderOrder = drawOrder.handle;
                this.styleHandle(handle);
                this.scene.add(handle); return handle;
            });
            this.editing?.setPoints(points);
        }
        points.forEach((point, index) => this.handles[index].position.fromArray(point));
    }

    pointLabel(index) {
        return this.shape && this.shape.length === 2 ? `(${Math.floor(index / this.shape[1])}, ${index % this.shape[1]})` : String(index);
    }

    setSelection(indices,active=indices.at(-1)) {
        this.selectedIndices=new Set(indices.filter(index=>Number.isInteger(index)&&index>=0&&index<this.handles.length));
        this.selected=this.selectedIndices.has(active)?active:(this.selectedIndices.values().next().value??-1);
        this.root.dataset.selectedPoint = String(this.selected);
        this.root.dataset.selectedPoints=JSON.stringify([...this.selectedIndices]);
        this.updateSelection(); this.render();
    }

    select(index,{toggle=false,preserve=false}={}) {
        index=Math.min(Math.max(index,0),this.handles.length-1);
        const indices=new Set(this.selectedIndices||[this.selected]);
        if(toggle){if(indices.has(index))indices.delete(index);else indices.add(index);}
        else if(!preserve||!indices.has(index)){indices.clear();indices.add(index);}
        this.setSelection([...indices],index);
    }

    getWeight() {
        return this.shape.length === 1 ? this.model.weights[this.selected] :
            this.model.weights[Math.floor(this.selected / this.shape[1])][this.selected % this.shape[1]];
    }

    updateSelection() {
        if (!this.model) return;
        this.editing?.updateSelection();
        this.handles.forEach((handle, i) => {
            handle.visible=this.handleCheck?.checked??true;
            const active=!this.readOnly && (this.selectedIndices?.has(i)??i===this.selected);
            handle.material=active ? this.selectedMaterial : this.pointMaterial;
            if (handle.userData.outline) handle.userData.outline.material=active ? this.selectedOutlineMaterial : this.outlineMaterial;
        });
    }

    editWeight(value) {
        this.editing?.editWeight(value);
    }

    updateGeometry(data) {
        const quality = data.quality || 'full';
        const positions = data.positions || new Float32Array(data.points.flat());
        const topology = this.topologies[quality] || this.topologies.full;
        const isSurface=Boolean(topology.triangles?.length);
        let geometry = this.geometryCache.get(quality);
        if (!geometry) {
            if(isSurface){
                geometry = new THREE.BufferGeometry();
                geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions.length), 3));
                geometry.setIndex(topology.triangles.flat());
            }else geometry=lineGeometry(positions);
            this.geometryCache.set(quality, geometry);
        }
        if(isSurface){
            if (geometry.attributes.position.array.length !== positions.length) throw new Error('Unexpected display buffer size.');
            geometry.attributes.position.array.set(positions);geometry.attributes.position.needsUpdate = true;
            geometry.computeVertexNormals(); // Display shading only.
        }else updateLinePositions(geometry,positions);
        if (!this.geometryObject) {
            if (isSurface) {
                this.geometryObject = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({color:colors.surface,metalness:.02,roughness:.52,side:THREE.DoubleSide,transparent:true,opacity:.98}));
            } else {
                this.geometryObject = wideLine(geometry,{color:colors.curve,width:2.6});
            }
            this.geometryObject.renderOrder=drawOrder.geometry;
            // Edited vertices can leave the original bounding sphere.
            this.geometryObject.frustumCulled = false;
            this.geometryObject.material.polygonOffset=true;this.geometryObject.material.polygonOffsetFactor=1;this.geometryObject.material.polygonOffsetUnits=1;
            this.scene.add(this.geometryObject);
        }
        this.geometryObject.geometry = geometry;
        this.displayQuality=quality;
        if(data.model || this.model) this.displayModel=clone(data.model || this.model);
        this.scientific?.geometryChanged();
        if(isSurface && this.edgeCheck) this.updateSurfaceEdges(positions,quality,topology.count);
    }

    updateSurfaceEdges(positions,quality,count) {
        if(!this.edgeCache) this.edgeCache=new Map();
        let cached=this.edgeCache.get(quality);
        if(!cached){
            cached=boundaryIndices(count).map(indices=>({indices,positions:new Float32Array(indices.length*3)}));
            this.edgeCache.set(quality,cached);
        }
        cached.forEach((edge,index)=>{
            edge.indices.forEach((vertex,i)=>edge.positions.set(positions.subarray(vertex*3,vertex*3+3),i*3));
            if(!edge.geometry) edge.geometry=lineGeometry(edge.positions);
            else updateLinePositions(edge.geometry,edge.positions);
            if(!this.surfaceEdges) this.surfaceEdges=[];
            if(!this.surfaceEdges[index]){
                const line=wideLine(edge.geometry,{width:2.6,depthTest:!this.is2D,order:1});
                this.surfaceEdges[index]=line;this.scene.add(line);
            }
            this.surfaceEdges[index].geometry=edge.geometry;
            this.surfaceEdges[index].visible=this.edgeCheck.checked;
        });
    }

    updateNet(lines) {
        if (lines) this.controlLines = lines;
        if (!this.net) {
            if (!this.controlLines) return;
            this.netIndices = [];
            for (const line of this.controlLines) {
                for (let i=1;i<line.length;i++) this.netIndices.push(line[i-1],line[i]);
            }
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(this.netIndices.length*3),3));
            this.net = new THREE.LineSegments(geometry,new THREE.LineBasicMaterial({color:colors.control,transparent:true,opacity:.9,depthTest:false,depthWrite:false}));
            this.net.renderOrder=drawOrder.controlPolygon;this.scene.add(this.net);
        }
        const attribute = this.net.geometry.attributes.position;
        this.netIndices.forEach((index,i) => this.handles[index].position.toArray(attribute.array,i*3));
        attribute.needsUpdate = true;
        this.net.frustumCulled = false;
        this.net.visible = this.netCheck.checked;
    }

    resize() {
        if (this.disposed) return;
        const width = Math.max(this.viewport.clientWidth, 1), height = Math.max(this.viewport.clientHeight, 1);
        this.renderer.setSize(width, height, false);
        const rectangle=this.plotRectangle(width,height);
        if (this.is2D) {
            const half = this.orthoHalf || 3, aspect = rectangle.width / rectangle.height;
            this.camera.left = -half * aspect; this.camera.right = half * aspect;
            this.camera.top = half; this.camera.bottom = -half;
        } else this.camera.aspect = rectangle.width / rectangle.height;
        this.camera.updateProjectionMatrix();
        this.controls.handleResize?.();
        if (this.navigation==='arcball' && !this.cameraAnimating) this.controls.setCamera(this.camera);
        this.render();
    }

    fit() {
        if (!this.handles.length) return;
        this.viewCube?.cancel(false);
        const box = this.options?.bounds?.() || new THREE.Box3().setFromPoints(this.handles.map(handle => handle.position));
        const center = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3());
        const extent = Math.max(size.length(), 1e-6);
        this.references?.setBounds(box,extent,this.is2D?this.viewMode:false);
        this.controls.target.copy(center);
        this.camera.near = extent * .0001; this.camera.far = extent * 1000;
        this.camera.zoom = 1;
        if (this.is2D) {
            const rectangle=this.plotRectangle();
            const aspect = Math.max(rectangle.width,1)/Math.max(rectangle.height,1);
            const plane=projectionPlanes[this.viewMode];
            this.orthoHalf = Math.max(size.getComponent(plane.vertical), size.getComponent(plane.horizontal) / aspect, extent * .2) * .67;
            this.camera.position.copy(center).add(new THREE.Vector3(...plane.direction).multiplyScalar(extent*3));
            this.camera.up.fromArray(plane.up);
        } else {
            this.camera.position.copy(center).add(new THREE.Vector3(1.3, -1.7, 1.5).normalize().multiplyScalar(extent * 1.8));
            this.camera.up.set(0, 0, 1);
        }
        this.camera.lookAt(center);this.handles.forEach(handle=>handle.quaternion.copy(this.is2D?this.camera.quaternion:new THREE.Quaternion())); this.resize();
        this.setNavigation(this.navigation,{target:center,cancelCube:false}); this.updateSelection();this.render();
    }

    setRay(event) {
        const bounds = this.renderer.domElement.getBoundingClientRect();
        const rectangle=this.plotRectangle(bounds.width,bounds.height);
        const x=event.clientX-bounds.left-rectangle.left,y=event.clientY-bounds.top-rectangle.top;
        this.pointer.set(x/rectangle.width*2-1,1-y/rectangle.height*2);
        this.raycaster.setFromCamera(this.pointer, this.camera);
        return x>=0 && y>=0 && x<=rectangle.width && y<=rectangle.height;
    }

    pickControlPoint(event) {
        if(!this.handles?.length||this.setRay(event)===false)return null;
        return this.raycaster.intersectObjects(this.handles.filter(h=>h.visible!==false),false)[0]?.object||null;
    }

    pointerDown(event) {
        if(this.disposed||event.button!==0)return;
        if(this.editing){this.editing.pointerDown(event);return;}
        // Selection is inspection in viewer mode; it never changes geometry.
        const handle=this.pickControlPoint(event);
        if(handle){event.preventDefault();event.stopImmediatePropagation();this.select(handle.userData.index);}
    }

    pointerMove(event) {this.editing?.pointerMove(event);}
    endDrag() {this.editing?.endDrag();}
    pointerUp(event) {this.editing?.pointerUp(event);}

    enqueue(action, immediate = false, quality = 'full') {
        if (this.readOnly) return;
        this.pending = {action, quality, model: clone(this.model), generation: this.localGeneration};
        this.pending.historyToken=this.editing?.history?.transaction?.token;
        this.editing?.updateHistory();
        this.options.onBusyChange?.(true);
        this.setStatus('Updating in Python…'); this.root.classList?.add('nrb-updating');
        if (immediate) this.flush();
        else if (!this.sendTimer) this.sendTimer = setTimeout(() => {this.sendTimer = null; this.flush();}, 0);
    }

    flush() {
        if (this.sendTimer) {clearTimeout(this.sendTimer); this.sendTimer = null;}
        if (this.disposed || !this.pending || this.inFlight) return;
        const request = this.pending; this.pending = null;
        request.revision = ++this.nextRevision;
        this.inFlight = request;
        this.root.dataset.requestedRevision = String(request.revision);
        if (this.options.evaluate) {
            this.send(request);
        } else if (this.options.onEdit) {
            this.options.onEdit({revision:request.revision,action:request.action,model:request.model});
        } else {
            this.inFlight = null; this.setStatus('The Python connection is unavailable.', true);
            this.options.onBusyChange?.(false);
        }
    }

    async send(request) {
        const controller = new AbortController();
        this.controller = controller;
        const timeout = setTimeout(() => controller.abort(), 30000);
        try {
            const packet={revision:request.revision,action:request.action,quality:request.quality,
                control_points:request.model.control_points,weights:request.model.weights};
            // A bridge cannot necessarily cancel Python work, so race a timeout
            // as well as providing the HTTP adapter with an AbortSignal.
            const aborted=new Promise((_,reject)=>{
                controller.signal.addEventListener('abort',()=>reject(new Error('Python evaluation timed out.')),{once:true});
            });
            const result=await Promise.race([this.options.evaluate(packet,controller.signal),aborted]);
            if (result.revision !== request.revision) throw new Error('Unexpected edit revision.');
            const quality = result.quality;
            if (!this.topologies[quality]) throw new Error('Unexpected display quality.');
            const positions = result.positions;
            if (!(positions instanceof Float32Array)) throw new Error('Invalid display data.');
            if (positions.length !== 3*this.topologies[quality].count) throw new Error('Incomplete display data.');
            if (!this.disposed) this.apply({revision:request.revision,model:request.model,quality,positions,error:null});
        } catch(error) {
            if (!this.disposed) this.apply({revision:request.revision,model:this.acceptedModel,error:error.message || 'Python is unavailable.'});
        } finally {
            clearTimeout(timeout);
        }
    }

    render() {
        if (this.disposed || this.renderFrame !== null) return;
        this.renderFrame = requestAnimationFrame(() => {
            this.renderFrame = null;
            if (!this.disposed) {
                if (this.navigation==='trackball' && !this.cameraAnimating) this.controls.update();
                const framed=Boolean(this.is2D&&this.axesCheck?.checked);
                const width=this.viewport.clientWidth,height=this.viewport.clientHeight,rectangle=this.plotRectangle(width,height);
                this.camera.updateMatrixWorld();
                this.options?.beforeRender?.();
                this.handles.forEach((handle,index)=>{
                    const radius=this.is2D?planarHandleRadius(this.camera,rectangle.height):spatialHandleRadius(this.camera,rectangle.height,handle.position);
                    handle.scale.setScalar(radius*(!this.readOnly&&(this.selectedIndices?.has(index)??index===this.selected)?1.15:1));
                });
                this.references?.updateTicks(this.camera,width,height,{enabled:framed&&Boolean(this.gridCheck?.checked),plane:this.viewMode,rectangle});
                renderLayeredScene(this.renderer,this.referenceScene,this.scene,this.camera,width,height,framed||this.options?.aspect?rectangle:null);
                this.planarAxes?.update(this.camera,width,height,{enabled:framed,plane:this.viewMode,rectangle});
                if(!this.is2D) this.viewCube?.update();
                this.options?.afterRender?.();
            }
        });
    }

    dispose() {
        if (this.disposed) return;
        this.disposed = true;
        this.root.removeEventListener('keydown',this.onInspectorKey);
        clearTimeout(this.sendTimer);
        if (this.renderFrame !== null) cancelAnimationFrame(this.renderFrame);
        this.controller?.abort();
        this.editing?.destroy();
        this.scientific?.destroy();
        this.viewCube?.destroy();
        this.planarAxes?.destroy();
        this.resizeObserver.disconnect(); this.removalObserver.disconnect(); this.controls.dispose();
        this.renderer.domElement.removeEventListener('pointerdown', this.onDown, true);
        this.renderer.domElement.removeEventListener('pointermove', this.onMove, true);
        this.renderer.domElement.removeEventListener('pointerup', this.onUp, true);
        this.renderer.domElement.removeEventListener('pointercancel', this.onUp, true);
        this.renderer.domElement.removeEventListener('lostpointercapture', this.onUp, true);
        this.renderer.domElement.removeEventListener('pointermove',this.onNavigationInput);
        this.renderer.domElement.removeEventListener('wheel',this.onNavigationInput);
        const geometries = new Set(this.geometryCache.values());
        if(this.edgeCache) for(const edges of this.edgeCache.values()) edges.forEach(edge=>geometries.add(edge.geometry));
        const materials = new Set([this.pointMaterial,this.selectedMaterial]);
        geometries.add(this.pointGeometry);
        if (this.outlineGeometry) geometries.add(this.outlineGeometry);
        if (this.outlineMaterial) materials.add(this.outlineMaterial);
        if (this.selectedOutlineMaterial) materials.add(this.selectedOutlineMaterial);
        this.scene.traverse(object => {
            if(object.geometry) geometries.add(object.geometry);
            if(object.material) materials.add(object.material);
        });
        this.referenceScene?.traverse(object=>{if(object.geometry)geometries.add(object.geometry);if(object.material)materials.add(object.material);});
        geometries.forEach(geometry=>geometry.dispose());
        materials.forEach(material=>{material.map?.dispose();material.dispose();});
        this.renderer.dispose(); this.renderer.forceContextLoss();
        viewers.delete(this.root.id);
    }
}

export function update(rootId, scene, options = {}) {
    const root = document.getElementById(rootId);
    if (!root || !scene) return null;
    try {
        // Keep the old experimental Dash entry point compatible. The viewer
        // itself depends only on callbacks, not on Dash stores or HTTP.
        options={...options};
        if (!options.evaluate && options.endpoint) options.evaluate=httpEvaluator(options.endpoint);
        if(!options.inspect && options.inspectEndpoint) options.inspect=async packet=>{
            const response=await fetch(options.inspectEndpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(packet)});
            const result=await response.json();if(!response.ok) throw new Error(result.error || 'Python inspection failed.');return result;
        };
        if (!options.onChange && options.stateStore) options.onChange=model=>window.dash_clientside.set_props(options.stateStore,{data:model});
        if (!options.readOnly && !options.evaluate && !options.onEdit && scene.edit_store) options.onEdit=edit=>window.dash_clientside.set_props(scene.edit_store,{data:edit});
        let viewer = viewers.get(rootId);
        if (viewer && viewer.root !== root) {viewer.dispose(); viewer = null;}
        if (!viewer) {viewer = new Explorer(root, scene, options); viewers.set(rootId, viewer);}
        else viewer.apply(scene);
        return {revision: scene.revision, mounted: true};
    } catch (error) {
        console.error('nurbspy web explorer:', error);
        root.replaceChildren(element('div', {class: 'nrb-failure', role: 'alert'}, `Could not initialize the geometry viewport: ${error.message}`));
        return {revision: scene.revision, error: error.message};
    }
}

export function editorState(rootId) {
    const viewer=viewers.get(rootId);
    if (!viewer) throw new Error('The editor is not mounted.');
    return {model:clone(viewer.acceptedModel),busy:Boolean(viewer.pending || viewer.inFlight || viewer.drag)};
}

export function destroy(rootId) {
    viewers.get(rootId)?.dispose();
}
