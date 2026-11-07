// Real editing operations with Three.js vectors; Python responses are controlled.
const assert=require('node:assert/strict');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
const assets=path.resolve(__dirname,'../../src/nurbspy/web_assets');
global.document={activeElement:null};
(async()=>{
    const THREE=await import(pathToFileURL(path.join(assets,'vendor/three.module.js')).href);
    const {EditingControls}=await import(pathToFileURL(path.join(assets,'editing.js')).href);
    function fixture(surface=false){
        const points=surface?[[[0,0],[1,1]],[[0,1],[0,1]],[[0,0],[0,0]]]:[[0,1],[0,1],[0,0]];
        const e={model:{control_points:points,weights:surface?[[1,1],[1,1]]:[1,1]},shape:surface?[2,2]:[2],
            selected:0,localGeneration:0,options:{},queued:[],controls:{enabled:true},
            enqueue(...args){this.queued.push(args);},updateNet(){},render(){},setStatus(text){this.status=text;},
            getWeight(){return surface?this.model.weights[0][0]:this.model.weights[this.selected];}};
        e.handles=Array.from({length:surface?4:2},(_,i)=>({position:new THREE.Vector3(...[0,1,2].map(d=>surface?points[d][Math.floor(i/2)][i%2]:points[d][i])),userData:{index:i}}));
        const edit=Object.create(EditingControls.prototype);Object.assign(edit,{owner:e,pointSelect:{},weight:{},weightSlider:{},coordinates:[{},{},{}],planeSelect:{value:'screen'},constraintHint:{}});
        e.updateSelection=()=>edit.updateSelection();e.select=index=>{e.selected=index;e.updateSelection();};
        return {e,edit};
    }
    for(const surface of [false,true]){
        const {e,edit}=fixture(surface);e.selected=surface?3:1;
        edit.editCoordinate(2,'1.23456789012345');
        assert.equal(e.handles[e.selected].position.z,1.23456789012345);
        assert.equal(surface?e.model.control_points[2][1][1]:e.model.control_points[2][1],1.23456789012345);
        assert.deepEqual(e.handles[0].position.toArray(),[0,0,0]);
        assert.deepEqual(e.queued,[['edit',true,'full']]);
        edit.editCoordinate(2,'1.23456789012345');assert.equal(e.queued.length,1);
    }
    console.log('PASS Numeric coordinates update curve/surface model layouts at full quality without redundant commits');
    {
        const {e,edit}=fixture();const original=JSON.stringify(e.model);
        for(const value of ['', ' ', 'NaN', 'Infinity'])edit.editCoordinate(0,value);
        assert.equal(JSON.stringify(e.model),original);assert.equal(e.queued.length,0);
        e.readOnly=true;edit.editCoordinate(0,'3');assert.equal(JSON.stringify(e.model),original);
    }
    console.log('PASS Invalid and read-only coordinate edits leave geometry unchanged');
    {
        const {e,edit}=fixture();e.select(1);assert.equal(edit.coordinates[0].value,1);
        document.activeElement=edit.coordinates[0];edit.coordinates[0].value='1.234e';e.updateSelection();
        assert.equal(edit.coordinates[0].value,'1.234e');document.activeElement=null;
    }
    console.log('PASS Selection populates coordinates without overwriting an unfinished focused field');
    {
        const {e,edit}=fixture();edit.planeSelect.value='normal';let release;
        const first=new Promise(resolve=>{release=resolve;});const requested=[];
        e.options.inspect=async request=>{requested.push(request);return request.revision===0?first:{revision:request.revision,normal:{direction:[1,0,0]}};};
        const running=edit.refreshNormal();e.localGeneration++;e.model.control_points[0][0]=2;
        release({revision:0,normal:{direction:[0,0,1]}});await running;await new Promise(setImmediate);
        assert.deepEqual(edit.normalDirection,[1,0,0]);assert.equal(edit.normalKey,'0:1');
        assert.equal(requested.length,2);assert.equal(requested[0].control_points[0][0],0);
    }
    console.log('PASS Stale Python normal responses are discarded and refreshed for the latest geometry');
    {
        const {e,edit}=fixture();edit.planeSelect.value='normal';
        e.options.inspect=async request=>({revision:request.revision,normal:{direction:null,reason:'Zero curvature.'}});
        await edit.refreshNormal();assert.equal(edit.normalDirection,null);assert.equal(edit.normalReason,'Zero curvature.');
        e.pickControlPoint=()=>e.handles[0];edit.pointerDown({button:0,preventDefault(){},stopImmediatePropagation(){}});
        assert.equal(e.drag,undefined);assert.equal(e.status,'Zero curvature.');
    }
    console.log('PASS Undefined normals explain the limitation and do not begin a drag');
    {
        const {e,edit}=fixture();edit.planeSelect.value='normal';edit.normalKey='0:0';edit.normalDirection=[0,0,1];
        e.camera=new THREE.PerspectiveCamera();e.camera.position.set(3,-4,3);e.camera.lookAt(0,0,0);e.camera.updateMatrixWorld();
        e.raycaster={ray:new THREE.Ray(e.camera.position.clone(),new THREE.Vector3().sub(e.camera.position).normalize())};
        e.pickControlPoint=()=>e.handles[0];e.setRay=()=>true;
        e.renderer={domElement:{style:{},setPointerCapture(){},hasPointerCapture(){return false;}}};
        edit.pointerDown({button:0,pointerId:7,preventDefault(){},stopImmediatePropagation(){}});
        assert.deepEqual(e.drag.axis.toArray(),[0,0,1]);edit.normalDirection=[1,0,0];
        e.raycaster.ray.direction.copy(new THREE.Vector3(0,0,1).sub(e.camera.position).normalize());
        edit.pointerMove({pointerId:7,preventDefault(){},stopImmediatePropagation(){}});
        const moved=e.handles[0].position;assert.ok(Math.abs(moved.z-1)<1e-12);
        assert.ok(Math.abs(moved.x)<1e-12&&Math.abs(moved.y)<1e-12);
        assert.deepEqual(e.drag.axis.toArray(),[0,0,1]);
    }
    console.log('PASS A normal drag stays on its initial axis even when the cached direction changes');
    console.log('6 editing checks passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
