// Mount the real shared UI with a minimal DOM, without a GPU or app framework.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const {pathToFileURL}=require('node:url');
const assets=path.resolve(__dirname,'../../src/nurbspy/web_assets');
class Element {
    constructor(tag){this.tag=tag;this.children=[];this.attributes={};this.listeners={};this.style={};this.hidden=false;
        const classes=new Set();this.classList={add:x=>classes.add(x),toggle:(x,on)=>on?classes.add(x):classes.delete(x)};}
    setAttribute(key,value){this.attributes[key]=String(value);if(key==='id')this.id=value;if(key==='hidden')this.hidden=true;}
    append(...children){for(const child of children){child.parentElement=this;this.children.push(child);}}
    appendChild(child){this.append(child);return child;}
    replaceChildren(...children){this.children=[];this.append(...children);}
    addEventListener(name,listener){(this.listeners[name]??=[]).push(listener);}
    dispatch(name,event={}){
        let stopped=false;const stop=event.stopPropagation;event.target??=this;
        event.stopPropagation=()=>{stopped=true;stop?.call(event);};
        for(let node=this;node;node=node.parentElement){
            for(const callback of node.listeners[name]||[])callback(event);
            if(stopped)break;
        }
    }
    click(){this.dispatch('click',{target:this});}
    focus(){document.activeElement=this;}
    contains(node){return this===node||this.children.some(child=>child.contains(node));}
    set value(value){this._value=value;}
    get value(){return this._value??this.children[0]?.attributes.value??'';}
}
global.document={createElement:tag=>new Element(tag),getElementById:()=>null,head:new Element('head')};
(async()=>{
    const {ScientificDisplay}=await import(pathToFileURL(path.join(assets,'science.js')).href);
    const {EditingControls}=await import(pathToFileURL(path.join(assets,'editing.js')).href);
    const {createInspector,syncInspector}=await import(pathToFileURL(path.join(assets,'panels.js')).href);
    const context={console,document,ScientificDisplay,EditingControls,createInspector,syncInspector,
        createPlanarAxes:()=>({destroy(){}})};
    vm.createContext(context);
    const source=fs.readFileSync(path.join(assets,'explorer.js'),'utf8').replace(/^import .*;\r?\n/gm,'').replace(/^export /gm,'');
    vm.runInContext(source+'\nglobalThis.Explorer=Explorer;',context);
    function mount(readOnly){
        const core=Object.create(context.Explorer.prototype);
        core.root=new Element('div');core.root.id=readOnly?'view':'edit';core.readOnly=readOnly;
        core.resize=()=>{};core.camera={unchanged:true};core.model={control_points:[[0],[0],[0]],weights:[1]};
        core.buildDOM({title:'Example',triangles:[],fields:{},colormaps:{},display:{}});return core;
    }
    const all=node=>[node,...node.children.flatMap(all)];
    const labels=core=>all(core.root).map(node=>node.attributes['aria-label']).filter(Boolean);
    const viewer=mount(true);
    assert.equal(viewer.editing,undefined);assert.deepEqual(Object.keys(viewer.panels),['display']);
    for(const label of ['Selected control point','Selected point colour','X coordinate','Y coordinate','Z coordinate','Drag constraint','Control point weight','Logarithmic weight slider'])assert.ok(!labels(viewer).includes(label));
    assert.ok(!all(viewer.root).some(node=>node.textContent==='Reset'));
    console.log('PASS Viewer mode mounts display/navigation without edit controls or Reset');
    const editor=mount(false);
    assert.ok(editor.editing instanceof EditingControls);
    for(const label of ['Selected control point','Selected point colour','X coordinate','Y coordinate','Z coordinate','Drag constraint','Control point weight','Logarithmic weight slider'])assert.ok(labels(editor).includes(label));
    viewer.render=()=>{};viewer.scientific.applyColors();
    viewer.scientific.controlColor.value='#8544cc';
    all(viewer.root).find(node=>node.textContent==='Restore default colours').click();
    assert.equal(viewer.scientific.controlColor.value,'#ff7f0e');
    for(const control of [editor.editing.pointSelect,editor.editing.weight,editor.editing.planeSelect])assert.ok(editor.panels.edit.sidebar.contains(control));
    for(const [axis,input] of editor.editing.coordinates.entries())assert.equal(input.parentElement.textContent,`${'XYZ'[axis]} coordinate`);
    assert.equal(editor.editing.planeSelect.children.find(node=>node.attributes.value==='normal').textContent,'Normal direction');
    assert.ok(!all(editor.panels.edit.sidebar).some(node=>node.tag==='small'));
    console.log('PASS Editor adds point, constraint, and weight controls only inside its Edit pane');
    for(const core of [viewer,editor]){
        const heading=core.root.children[0],workspace=core.root.children[1];
        assert.ok(heading.contains(core.navigationSelect));
        for(const panel of Object.values(core.panels)){
            assert.ok(heading.contains(panel.toggle));assert.ok(!workspace.contains(panel.toggle));
            assert.equal(panel.toggle.textContent,panel.kind==='display'?'Display':'Edit');
            assert.equal(panel.toggle.attributes['aria-controls'],panel.sidebar.id);
        }
        assert.equal(core.navigationSelect.tag,'select');
        assert.deepEqual(core.navigationSelect.children.map(node=>node.attributes.value),['arcball','orbit','trackball']);
        let changed;core.setNavigation=value=>{changed=value;};core.navigationSelect.value='trackball';
        core.navigationSelect.dispatch('change');assert.equal(changed,'trackball');
    }
    console.log('PASS Horizontal header buttons and native navigation dropdown work in both modes');
    const model=JSON.stringify(editor.model),camera=editor.camera;
    editor.panels.display.toggle.click();assert.equal(editor.activePanel,'display');
    assert.equal(editor.panels.display.toggle.attributes['aria-expanded'],'true');
    editor.panels.edit.toggle.click();assert.equal(editor.activePanel,'edit');assert.equal(editor.panels.display.sidebar.hidden,true);
    assert.equal(editor.panels.display.toggle.attributes['aria-expanded'],'false');
    assert.equal(editor.panels.edit.toggle.attributes['aria-expanded'],'true');
    editor.editing.weight.value='2.5';editor.panels.edit.toggle.click();assert.equal(editor.activePanel,null);
    assert.equal(editor.panels.edit.sidebar.hidden,true);assert.equal(editor.panels.edit.toggle.attributes['aria-expanded'],'false');
    editor.panels.display.toggle.click();editor.panels.edit.toggle.click();assert.equal(editor.editing.weight.value,'2.5');
    assert.equal(JSON.stringify(editor.model),model);assert.equal(editor.camera,camera);
    console.log('PASS Inspector switching preserves form, geometry, and camera state');
    editor.panels.edit.sidebar.children[0].children.find(node=>node.tag==='button').click();
    assert.equal(editor.activePanel,null);assert.equal(document.activeElement,editor.panels.edit.toggle);
    assert.equal(editor.editing.weight.value,'2.5');editor.panels.edit.toggle.click();
    console.log('PASS Pane close returns keyboard focus to its header button and retains form values');
    let prevented=0,stopped=0;
    editor.panels.edit.sidebar.dispatch('keydown',{key:'Escape',preventDefault(){prevented++;},stopPropagation(){stopped++;}});
    assert.equal(editor.activePanel,null);assert.equal(prevented,1);assert.equal(stopped,1);
    console.log('PASS Escape closes the pane and is not forwarded to the desktop host');
    viewer.panels.display.toggle.click();let rootPrevented=0,rootStopped=0;
    const escape={key:'Escape',preventDefault(){rootPrevented++;},stopPropagation(){rootStopped++;}};
    viewer.root.dispatch('keydown',escape);assert.equal(viewer.activePanel,null);
    assert.equal(rootPrevented,1);assert.equal(rootStopped,1);assert.equal(document.activeElement,viewer.panels.display.toggle);
    viewer.root.dispatch('keydown',escape);assert.equal(rootStopped,1);
    console.log('PASS Escape anywhere in the widget closes an open pane and otherwise reaches its host');
    editor.editing.planeSelect.value='xz';editor.is2D=true;editor.viewMode='yz';editor.editing.syncProjection();
    assert.equal(editor.editing.planeSelect.disabled,true);assert.equal(editor.editing.planeSelect.value,'yz');
    editor.is2D=false;editor.editing.syncProjection();assert.equal(editor.editing.planeSelect.value,'xz');
    assert.equal(editor.editing.planeSelect.disabled,false);
    console.log('PASS Projection changes restore the editor drag constraint when returning to 3D');
    console.log('8 capability checks passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
