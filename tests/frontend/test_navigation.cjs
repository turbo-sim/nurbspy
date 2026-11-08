// Run with Node 22+; uses the vendored Three.js math and stubbed browser controls.
// node tests/frontend/test_navigation.cjs [web_assets_directory] [three.module.js]
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {pathToFileURL} = require('node:url');
const assets = process.argv[2] || path.resolve(__dirname, '../../src/nurbspy/web_assets');
const threePath = process.argv[3] || path.join(assets, 'vendor/three.module.js');
let checks = 0;
function check(name, test) {test(); checks++; console.log('PASS', name);}
function near(actual, expected, tolerance=1e-10) {
    assert.ok(Math.abs(actual-expected) <= tolerance, `${actual} != ${expected}`);
}
function nearVector(actual, expected) {
    actual.forEach((value,index)=>near(value,expected[index]));
}

(async()=>{
    const THREE = await import(pathToFileURL(threePath).href);
    const cubeSource = fs.readFileSync(path.join(assets,'viewcube.js'),'utf8')
        .replace(/export function /g,'function ').replace(/export \{[^}]+\};?/g,'');
    const cubeContext = {console};
    vm.createContext(cubeContext);
    vm.runInContext(cubeSource+'\nglobalThis.math={basis,targetBasis,quaternion,fromQuaternion,interpolate};',cubeContext);
    const math = cubeContext.math;
    const vec = array=>new THREE.Vector3(...array);
    const directions=[];
    for(const x of [-1,0,1]) for(const y of [-1,0,1]) for(const z of [-1,0,1]) {
        if(x||y||z) directions.push([x,y,z]);
    }
    check('All 26 ViewCube face, edge, and corner directions form right-handed camera frames',()=>{
        for(const direction of directions) for(const up of [[0,0,1],[0,1,0]]) {
            const frame=math.targetBasis(direction,up);
            const right=vec(frame.right), vertical=vec(frame.up), back=vec(frame.back);
            near(right.length(),1);near(vertical.length(),1);near(back.length(),1);
            near(right.dot(vertical),0);near(right.dot(back),0);near(vertical.dot(back),0);
            near(right.clone().cross(vertical).dot(back),1);
            nearVector(back.toArray(),vec(direction).normalize().toArray());
        }
    });
    check('ViewCube quaternion encoding agrees with native Three.js camera rotations',()=>{
        for(const direction of directions) {
            const frame=math.targetBasis(direction);
            const matrix=new THREE.Matrix4().makeBasis(vec(frame.right),vec(frame.up),vec(frame.back));
            const expected=new THREE.Quaternion().setFromRotationMatrix(matrix);
            const actual=new THREE.Quaternion(...math.quaternion(frame));
            near(Math.abs(actual.dot(expected)),1);
            const restored=math.fromQuaternion(math.quaternion(frame));
            for(const axis of ['right','up','back']) nearVector(restored[axis],frame[axis]);
        }
    });
    check('Animated transitions keep orthonormal orientation even between opposite cube faces',()=>{
        const start=math.quaternion(math.targetBasis([1,0,0]));
        const end=math.quaternion(math.targetBasis([-1,0,0]));
        for(const fraction of [0,.1,.25,.5,.75,.9,1]) {
            const q=math.interpolate(start,end,fraction), frame=math.fromQuaternion(q);
            near(Math.hypot(...q),1);
            near(vec(frame.right).dot(vec(frame.up)),0);
            near(vec(frame.right).clone().cross(vec(frame.up)).dot(vec(frame.back)),1);
        }
    });
    class Element {
        constructor() {this.style={};this.classList={add(){}};this.listeners=[];}
        setAttribute() {}
        addEventListener(...args) {this.listeners.push(args);}
        removeEventListener() {}
    }
    class Renderer {
        constructor() {this.domElement=new Element();}
        setPixelRatio() {}
        setSize() {}
    }
    class Control {
        constructor(camera,domElement) {
            this.object=camera;this.domElement=domElement;this.target=new THREE.Vector3();
            this.constructedUp=camera.up.clone();this.disposed=false;this.events=[];
            camera.lookAt(this.target);
        }
        update() {this.updates=(this.updates||0)+1;this.object.lookAt(this.target);}
        addEventListener(...event) {this.events.push(event);}
        dispose() {this.disposed=true;}
    }
    class OrbitControls extends Control {}
    class TrackballControls extends Control {handleResize(){this.resizeCount=(this.resizeCount||0)+1;}}
    class ArcballControls extends Control {
        setCamera(camera) {
            this.cameraSyncCount=(this.cameraSyncCount||0)+1;
            this.targetAtCameraSync=this.target.clone();camera.lookAt(this.target);
        }
    }
    const nativeThree={...THREE,WebGLRenderer:Renderer};
    const display=await import(pathToFileURL(path.join(assets,'display.js')).href);
    const {EditingControls}=await import(pathToFileURL(path.join(assets,'editing.js')).href);
    const explorerContext={console,THREE:nativeThree,projectionPlanes:display.projectionPlanes,planarHandleRadius:display.planarHandleRadius,spatialHandleRadius:display.spatialHandleRadius,renderLayeredScene:display.renderLayeredScene,OrbitControls,TrackballControls,ArcballControls,
        ResizeObserver:class {observe() {}},
        createViewCube:()=>({setHome(){},cancel(){this.cancelCount=(this.cancelCount||0)+1;}}),
        window:{devicePixelRatio:1}};
    const explorerSource=fs.readFileSync(path.join(assets,'explorer.js'),'utf8')
        .replace(/^import .*;\r?\n/gm,'').replace(/^export /gm,'');
    vm.createContext(explorerContext);
    vm.runInContext(explorerSource+'\nglobalThis.Explorer=Explorer;',explorerContext);
    const Explorer=explorerContext.Explorer;
    function editor(dimension=3) {
        const e=Object.create(Explorer.prototype);
        Object.assign(e,{root:{dataset:{},classList:{toggle(){}}},viewport:{prepend(){},clientWidth:800,clientHeight:450},
            navigationSelect:{value:''},render(){this.drawRequests=(this.drawRequests||0)+1;}});
        e.buildScene({dimension});
        e.shape=[2];e.model={control_points:[[0,1],[0,1],[0,0]],weights:[1,1]};
        e.handles=[new THREE.Mesh(e.pointGeometry,e.pointMaterial),new THREE.Mesh(e.pointGeometry,e.pointMaterial)];
        e.handles[0].position.set(0,0,0);e.handles[1].position.set(1,1,0);
        e.selected=0;
        e.editing=Object.create(EditingControls.prototype);e.editing.owner=e;
        e.editing.pointSelect={};e.editing.weight={};e.editing.weightSlider={};e.editing.position={};e.editing.planeSelect={};e.editing.constraintHint={};
        return e;
    }
    check('3D starts in Arcball with Z-up configured before control construction',()=>{
        const e=editor();assert.equal(e.navigation,'arcball');
        nearVector(e.controls.constructedUp.toArray(),[0,0,1]);
        assert.equal(e.controls.enableAnimations,false);assert.equal(e.controls.enableGizmos,false);
        assert.ok(e.viewCube);assert.equal(e.root.dataset.navigation,'arcball');
    });
    check('Changing navigation preserves framing and disposes previous event handlers',()=>{
        const e=editor();const target=new THREE.Vector3(1.2,-.4,.6);
        e.camera.position.set(5,-7,3);e.camera.up.set(.1,.2,1).normalize();e.camera.zoom=1.8;
        e.controls.target.copy(target);e.camera.lookAt(target);
        const position=e.camera.position.clone(),up=e.camera.up.clone(),quaternion=e.camera.quaternion.clone();
        for(const mode of ['trackball','orbit','arcball','orbit','trackball']) {
            const previous=e.controls;e.setNavigation(mode);
            assert.ok(previous.disposed);assert.notEqual(e.controls,previous);
            nearVector(e.camera.position.toArray(),position.toArray());
            nearVector(e.camera.up.toArray(),up.toArray());near(e.camera.zoom,1.8);
            nearVector(e.controls.target.toArray(),target.toArray());
            near(Math.abs(e.camera.quaternion.dot(quaternion)),1);
            assert.equal(e.root.dataset.navigation,mode);assert.equal(e.navigationSelect.value,mode);
            if(mode==='trackball') {assert.equal(e.controls.staticMoving,true);assert.ok(e.controls.resizeCount);}
            if(mode==='arcball') nearVector(e.controls.targetAtCameraSync.toArray(),target.toArray());
        }
    });
    check('Orthographic projections keep planar navigation even when free rotation is requested',()=>{
        const e=editor();e.setViewMode('xy');
        for(const mode of ['trackball','arcball','orbit']) {
            e.setNavigation(mode);assert.equal(e.navigation,'orbit');assert.equal(e.controls.enableRotate,false);
            nearVector(e.controls.constructedUp.toArray(),[0,1,0]);assert.ok(e.viewCube);
        }
    });
    check('Invalid navigation choices do not dispose the working controller',()=>{
        const e=editor(),previous=e.controls;
        assert.throws(()=>e.setNavigation('unknown'),/Unknown navigation/);
        assert.equal(e.controls,previous);assert.equal(previous.disposed,false);
    });
    check('ViewCube animation preserves pan and zoom and rebases navigation only on completion',()=>{
        const e=editor();e.setNavigation('trackball');e.camera.zoom=1.8;
        const previous=e.controls,target=[1.2,-.4,.6];
        const first={position:[4,-6,2],up:[0,0,1],target};
        e.writeView(first,false);assert.equal(e.controls,previous);assert.equal(e.cameraAnimating,true);
        nearVector(e.controls.target.toArray(),target);near(e.camera.zoom,1.8);
        const last={position:[1.2,-.4,8.6],up:[0,1,0],target};
        const cancels=e.viewCube.cancelCount;
        e.writeView(last,true);
        assert.ok(previous.disposed);assert.notEqual(e.controls,previous);
        assert.equal(e.cameraAnimating,false);assert.equal(e.viewCube.cancelCount,cancels);
        nearVector(e.camera.position.toArray(),last.position);nearVector(e.camera.up.toArray(),last.up);
        nearVector(e.controls.target.toArray(),target);near(e.camera.zoom,1.8);
        nearVector(new THREE.Vector3(0,0,-1).applyQuaternion(e.camera.quaternion).toArray(),[0,0,-1]);
    });
    check('Control handles use planar outlined circles in 2D and a rounded material in 3D',()=>{
        const flat=editor(),spatial=editor();flat.setViewMode('xy');
        assert.equal(flat.pointGeometry.type,'CircleGeometry');assert.equal(flat.outlineGeometry.type,'RingGeometry');
        assert.equal(spatial.pointGeometry.type,'SphereGeometry');assert.equal(spatial.pointMaterial.type,'MeshPhysicalMaterial');
        assert.ok(spatial.pointMaterial.clearcoat>0);assert.equal(spatial.outlineGeometry,null);
    });
    check('View mode switches preserve native model coordinates and restore the spatial controller',()=>{
        const e=editor(3);e.setNavigation('trackball');
        e.surfaceEdges=[{material:{depthTest:true}}];
        const state=JSON.stringify(e.model);
        e.setViewMode('xy');assert.equal(e.is2D,true);assert.equal(e.camera.isOrthographicCamera,true);
        assert.equal(e.controls.enableRotate,false);assert.equal(JSON.stringify(e.model),state);
        assert.equal(e.surfaceEdges[0].material.depthTest,false);
        assert.equal(e.pointGeometry.type,'CircleGeometry');
        e.setViewMode('3d');assert.equal(e.camera.isPerspectiveCamera,true);assert.equal(e.navigation,'trackball');
        assert.equal(e.surfaceEdges[0].material.depthTest,true);
        assert.equal(e.pointGeometry.type,'SphereGeometry');assert.equal(JSON.stringify(e.model),state);
    });
    check('All orthographic modes use a right-handed camera basis and preserve spatial data',()=>{
        const e=editor(),state=JSON.stringify(e.model);
        for(const mode of ['xy','yz','zx']){
            e.setViewMode(mode);const plane=display.projectionPlanes[mode];
            assert.equal(e.is2D,true);assert.equal(e.dimension,3);assert.equal(e.controls.enableRotate,false);
            nearVector(e.camera.getWorldDirection(new THREE.Vector3()).toArray(),plane.direction.map(x=>-x));
            nearVector(new THREE.Vector3(0,1,0).applyQuaternion(e.camera.quaternion).toArray(),plane.up);
            assert.equal(JSON.stringify(e.model),state);
        }
    });
    check('Dragging a spatial control point in X–Y view preserves its native Z coordinate',()=>{
        const e=editor(3);e.model.control_points[2]=[2,3];e.handles[0].position.z=2;
        e.setViewMode('xy');
        Object.assign(e,{localGeneration:0,setRay(){return true;},updateNet(){},enqueue(){},
            raycaster:{ray:new THREE.Ray(new THREE.Vector3(1.5,-.5,10),new THREE.Vector3(0,0,-1))},
            drag:{id:7,index:0,start:new THREE.Vector3(0,0,2),hit:new THREE.Vector3(0,0,2),
                plane:new THREE.Plane(new THREE.Vector3(0,0,1),-2)}});
        e.pointerMove({pointerId:7,preventDefault(){},stopImmediatePropagation(){}});
        nearVector(e.handles[0].position.toArray(),[1.5,-.5,2]);
        assert.equal(e.model.control_points[2][0],2);
        assert.equal(e.model.control_points[0][0],1.5);assert.equal(e.model.control_points[1][0],-.5);
    });
    check('Viewers keep normal point colours on load and inspection in every projection',()=>{
        const e=editor(),state=JSON.stringify(e.model);delete e.editing;e.readOnly=true;
        e.handles[1].userData.index=1;e.pickControlPoint=()=>e.handles[1];
        for(const mode of ['3d','xy','yz','zx','3d']){
            e.setViewMode(mode);e.select(0);
            for(const handle of e.handles){
                assert.equal(handle.material,e.pointMaterial);
                if(handle.userData.outline)assert.equal(handle.userData.outline.material,e.outlineMaterial);
            }
            e.pointerDown({button:0,preventDefault(){},stopImmediatePropagation(){}});
            assert.equal(e.selected,1);assert.equal(e.drag,undefined);assert.equal(JSON.stringify(e.model),state);
            for(const handle of e.handles){
                assert.equal(handle.material,e.pointMaterial);
                if(handle.userData.outline)assert.equal(handle.userData.outline.material,e.outlineMaterial);
            }
        }
    });
    check('Only editors highlight and enlarge the active point in spatial and planar views',()=>{
        let draw;
        explorerContext.requestAnimationFrame=callback=>{draw=callback;return 1;};
        explorerContext.renderLayeredScene=()=>{};
        for(const readOnly of [false,true])for(const mode of ['3d','xy','yz','zx']){
            const e=editor();e.readOnly=readOnly;if(readOnly)delete e.editing;
            e.setViewMode(mode);e.select(1);e.renderFrame=null;e.viewCube=null;
            Explorer.prototype.render.call(e);draw();
            assert.equal(e.handles[1].material,readOnly?e.pointMaterial:e.selectedMaterial);
            if(e.is2D)assert.equal(e.handles[1].userData.outline.material,readOnly?e.outlineMaterial:e.selectedOutlineMaterial);
            e.handles.forEach((handle,index)=>{
                const radius=e.is2D?display.planarHandleRadius(e.camera,450):display.spatialHandleRadius(e.camera,450,handle.position);
                near(handle.scale.x/radius,!readOnly&&index===1?1.15:1);
            });
        }
    });
    console.log(`${checks} navigation checks passed`);
})().catch(error=>{console.error(error);process.exitCode=1;});
