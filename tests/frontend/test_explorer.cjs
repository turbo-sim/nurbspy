// Dependency-free controller/buffer tests: node tests/frontend/test_explorer.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const sourcePath = process.argv[2] || path.resolve(__dirname, '../../src/nurbspy/web_assets/explorer.js');
const source = fs.readFileSync(sourcePath,'utf8').replace(/^import .*;\r?\n/gm,'').replace(/^export /gm,'');
let nextId=0;
const frames=new Map(),timers=new Map(),published=[];
class BufferGeometry {
    constructor(){this.attributes={};this.index=null;this.disposed=false;}
    setAttribute(key,value){this.attributes[key]=value;}
    setIndex(indices){this.index=indices;}
    computeVertexNormals(){this.normalUpdates=(this.normalUpdates||0)+1;}
    dispose(){this.disposed=true;}
}
class BufferAttribute {constructor(array,size){this.array=array;this.itemSize=size;}}
class Material {constructor(options){this.options=options;}dispose(){this.disposed=true;}}
class Object3D {constructor(geometry,material){this.geometry=geometry;this.material=material;}}
const context={console,Float32Array,Map,Set,AbortController,
    renderLayeredScene:(renderer,background,scene,camera)=>renderer.render(scene,camera),
    THREE:{BufferGeometry,BufferAttribute,MeshStandardMaterial:Material,LineBasicMaterial:Material,
           Mesh:Object3D,Line:Object3D,LineSegments:Object3D,DoubleSide:2},
    requestAnimationFrame:fn=>{const id=++nextId;frames.set(id,fn);return id;},
    cancelAnimationFrame:id=>frames.delete(id),
    setTimeout:fn=>{const id=++nextId;timers.set(id,fn);return id;},clearTimeout:id=>timers.delete(id),
    window:{dash_clientside:{set_props:(id,value)=>published.push({id,value})}},
};
vm.createContext(context);vm.runInContext(source+'\nglobalThis.Explorer=Explorer;globalThis.httpEvaluator=httpEvaluator;',context);
const Explorer=context.Explorer;
let checks=0;
function check(name,test){test();checks++;console.log('PASS',name);}
function editor(){
    const object=Object.create(Explorer.prototype);
    Object.assign(object,{root:{id:'example',dataset:{}},options:{evaluate:context.httpEvaluator('/edit'),onChange:model=>published.push({id:'state',value:{data:model}})},
        lastRevision:0,nextRevision:0,localGeneration:0,pending:null,inFlight:null,sendTimer:null,drag:null,
        model:{control_points:[[0,1],[0,1]],weights:[1,1]},shape:[2],
        badge:{classList:{add(){},toggle(){}}},setStatus(){},render(){},updateSelection(){},
        moveHandles(){this.handleMoves=(this.handleMoves||0)+1;},updateNet(){},updateGeometry(){},
        send(request){this.sent.push(request);},sent:[]});
    object.acceptedModel=JSON.parse(JSON.stringify(object.model));
    return object;
}
check('Rapid preview events coalesce to the latest coordinates',()=>{
    const e=editor();e.enqueue('edit',false,'preview');
    e.model.control_points[0][0]=2;e.localGeneration++;e.enqueue('edit',false,'preview');
    assert.equal(e.sent.length,0);e.flush();assert.equal(e.sent.length,1);
    assert.equal(e.sent[0].model.control_points[0][0],2);
});
check('Only one evaluation is in flight',()=>{
    const e=editor();e.enqueue('edit',true,'preview');
    e.model.control_points[0][0]=3;e.localGeneration++;e.enqueue('edit',true,'preview');
    assert.equal(e.sent.length,1);assert.equal(e.pending.model.control_points[0][0],3);
});
check('Preview acknowledgement does not overwrite newer local input',()=>{
    const e=editor();e.enqueue('edit',true,'preview');const first=e.sent[0];
    e.model.control_points[0][0]=3;e.localGeneration++;e.enqueue('edit',true,'full');
    e.apply({revision:1,model:first.model,quality:'preview',error:null});
    assert.equal(e.model.control_points[0][0],3);assert.equal(e.handleMoves,undefined);
    assert.equal(e.sent.length,2);assert.equal(e.sent[1].quality,'full');
});
check('Full accepted model is published only after the pending queue drains',()=>{
    published.length=0;const e=editor();e.enqueue('edit',true,'preview');
    e.apply({revision:1,model:e.sent[0].model,quality:'preview',error:null});assert.equal(published.length,0);
    e.enqueue('edit',true,'full');e.apply({revision:2,model:e.sent[1].model,quality:'full',error:null});
    assert.equal(published.length,1);assert.equal(published[0].id,'state');
});
check('Stale responses do not replace accepted geometry',()=>{
    const e=editor();e.lastRevision=4;e.apply({revision:2,model:{},error:null});
    assert.equal(e.lastRevision,4);assert.equal(e.handleMoves,undefined);
});
check('Rejected edit acknowledges the request and restores accepted coordinates',()=>{
    const e=editor();e.model.control_points[0][0]=9;e.enqueue('edit',true,'full');
    e.apply({revision:1,model:e.acceptedModel,error:'invalid'});
    assert.equal(e.model.control_points[0][0],0);assert.equal(e.inFlight,null);
});
check('Rendering is coalesced into one animation-frame callback',()=>{
    frames.clear();const e={renderFrame:null,disposed:false,scene:{},camera:{updateMatrixWorld(){}},handles:[],viewport:{clientWidth:800,clientHeight:450},plotRectangle(){return {height:450};},renderer:{render(){e.draws++;}},draws:0};
    Explorer.prototype.render.call(e);Explorer.prototype.render.call(e);assert.equal(frames.size,1);
    const callback=frames.values().next().value;frames.clear();callback();assert.equal(e.draws,1);assert.equal(e.renderFrame,null);
});
check('Surface position buffers, mesh and material are reused',()=>{
    const e={geometryCache:new Map(),topologies:{full:{triangles:[[0,1,2]],count:3}},scene:{add(){}}};
    const first={quality:'full',positions:new Float32Array([0,0,0,1,0,0,0,1,0])};
    Explorer.prototype.updateGeometry.call(e,first);
    const mesh=e.geometryObject,geometry=mesh.geometry,material=mesh.material,array=geometry.attributes.position.array;
    first.positions[2]=2;Explorer.prototype.updateGeometry.call(e,first);
    assert.equal(e.geometryObject,mesh);assert.equal(mesh.geometry,geometry);assert.equal(mesh.material,material);
    assert.equal(geometry.attributes.position.array,array);assert.equal(array[2],2);assert.equal(geometry.disposed,false);
});
check('Preview/full topology switches reuse cached buffers',()=>{
    const e={geometryCache:new Map(),topologies:{full:{triangles:[[0,1,2]],count:3},preview:{triangles:[[0,1,2]],count:3}},scene:{add(){}}};
    const data={quality:'full',positions:new Float32Array(9)};Explorer.prototype.updateGeometry.call(e,data);
    const full=e.geometryObject.geometry;
    data.quality='preview';Explorer.prototype.updateGeometry.call(e,data);assert.notEqual(e.geometryObject.geometry,full);
    data.quality='full';Explorer.prototype.updateGeometry.call(e,data);assert.equal(e.geometryObject.geometry,full);
});
check('Control net updates preserve line geometry and material',()=>{
    const vector={toArray(array,offset){array.set([1,2,3],offset);}};
    const e={handles:[{position:vector},{position:vector}],netCheck:{checked:true},scene:{add(){}}};
    Explorer.prototype.updateNet.call(e,[[0,1]]);const net=e.net,geometry=net.geometry;
    Explorer.prototype.updateNet.call(e);assert.equal(e.net,net);assert.equal(e.net.geometry,geometry);
    assert.equal(geometry.attributes.position.array[2],3);
});
async function protocolCheck(name, makeResponse, verify) {
    const e=editor();e.topologies={full:{count:2}};
    const request={revision:1,action:'edit',quality:'full',model:e.model};
    let applied,packet;
    e.apply=data=>{applied=data;};
    context.fetch=async(url,options)=>{packet=JSON.parse(options.body);return makeResponse();};
    await Explorer.prototype.send.call(e,request);
    verify(applied,packet);checks++;console.log('PASS',name);
}
check('Read-only interaction never queues an edit',()=>{
    const e=editor();e.readOnly=true;e.enqueue('edit',true,'full');
    assert.equal(e.sent.length,0);assert.equal(e.pending,null);
    e.setRay=()=>{throw new Error('Read-only handles must not capture pointer input');};
    e.pointerDown({button:0});e.pointerMove({});
});
check('Busy state brackets a committed Python update',()=>{
    const e=editor(),states=[];e.options.onBusyChange=busy=>states.push(busy);
    e.enqueue('edit',true,'full');assert.equal(states.at(-1),true);
    e.apply({revision:1,model:e.sent[0].model,quality:'full',error:null});
    assert.equal(states.at(-1),false);
});
(async()=>{
    await protocolCheck('Compact transport decodes binary vertices without resending model metadata',
        ()=>new Response(new Float32Array([0,0,0,1,1,0]).buffer,{headers:{
            'X-Nurbspy-Revision':'1','X-Nurbspy-Quality':'full'}}),
        (data,packet)=>{
            assert.equal(data.error,null);assert.equal(data.positions.length,6);
            assert.deepEqual(Object.keys(packet).sort(),['action','control_points','quality','revision','weights']);
        });
    await protocolCheck('HTTP validation errors reach the rollback path',
        ()=>new Response(JSON.stringify({error:'Invalid weights'}),{status:422}),
        data=>assert.equal(data.error,'Invalid weights'));
    await protocolCheck('A mismatched binary-response revision is rejected',
        ()=>new Response(new Float32Array(6).buffer,{headers:{
            'X-Nurbspy-Revision':'9','X-Nurbspy-Quality':'full'}}),
        data=>assert.equal(data.error,'Unexpected edit revision.'));
    await protocolCheck('Truncated binary responses are rejected',
        ()=>new Response(new Float32Array(3).buffer,{headers:{
            'X-Nurbspy-Revision':'1','X-Nurbspy-Quality':'full'}}),
        data=>assert.equal(data.error,'Incomplete display data.'));
    const native=editor();native.topologies={full:{count:2}};
    let accepted;
    native.apply=data=>{accepted=data;};
    native.options.evaluate=async packet=>({revision:packet.revision,quality:packet.quality,positions:new Float32Array(6)});
    await Explorer.prototype.send.call(native,{revision:4,action:'edit',quality:'full',model:native.model});
    assert.equal(accepted.revision,4);assert.equal(accepted.error,null);
    checks++;console.log('PASS Direct native bridge uses the same controller without HTTP or Dash');
    native.options.evaluate=()=>new Promise(()=>{});
    const pending=Explorer.prototype.send.call(native,{revision:5,action:'edit',quality:'full',model:native.model});
    native.controller.abort();await pending;
    assert.equal(accepted.error,'Python evaluation timed out.');
    checks++;console.log('PASS An unresponsive native bridge can time out without blocking the queue');
    console.log(`${checks} frontend behavior checks passed`);
})().catch(error=>{console.error(error);process.exitCode=1;});
