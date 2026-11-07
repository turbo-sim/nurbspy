// Actual Three.js display geometry contracts, without a browser or GPU.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const {pathToFileURL}=require('node:url');
const assets=path.resolve(__dirname,'../../src/nurbspy/web_assets');
let checks=0;
function check(name,test){test();checks++;console.log('PASS',name);}
(async()=>{
    const THREE=await import(pathToFileURL(path.join(assets,'vendor/three.module.js')).href);
    const display=await import(pathToFileURL(path.join(assets,'display.js')).href);
    check('Wide curves retain their GPU attributes when sampled positions change',()=>{
        const values=new Float32Array([0,0,0,1,2,0,3,4,0]);
        const geometry=display.lineGeometry(values),buffer=geometry.attributes.instanceStart.data;
        values[4]=8;display.updateLinePositions(geometry,values);
        assert.equal(geometry.attributes.instanceStart.data,buffer);
        assert.equal(geometry.attributes.instanceEnd.getY(0),8);
        assert.equal(geometry.attributes.instanceStart.getY(1),8);
        assert.equal(geometry.attributes.instanceEnd.getX(1),3);
        assert.throws(()=>display.updateLinePositions(geometry,new Float32Array(6)),/sample count/);
    });
    check('Line thickness uses CSS pixels and survives viewport changes',()=>{
        const geometry=display.lineGeometry(new Float32Array([0,0,0,1,1,0]));
        const line=display.wideLine(geometry,{width:2.6,depthTest:false,order:1});
        assert.equal(line.material.linewidth,2.6);assert.equal(line.material.worldUnits,false);
        assert.equal(line.material.depthTest,false);assert.equal(line.renderOrder,1);
        assert.equal(line.frustumCulled,false);assert.equal(typeof line.onBeforeRender,'function');
    });
    check('Repeated scalar colouring reuses the curve GPU colour buffer',()=>{
        const geometry=display.lineGeometry(new Float32Array([0,0,0,1,1,0,2,0,0]));
        const colors=new Float32Array([0,0,0,.5,.5,.5,1,1,1]);
        display.updateLineColors(geometry,colors);const buffer=geometry.attributes.instanceColorStart.data;
        colors[3]=.25;display.updateLineColors(geometry,colors);
        assert.equal(geometry.attributes.instanceColorStart.data,buffer);
        assert.equal(geometry.attributes.instanceColorEnd.getX(0),.25);
        assert.equal(geometry.attributes.instanceColorStart.getX(1),.25);
    });
    check('Surface boundaries follow the native tensor-product sampling order',()=>{
        assert.deepEqual(display.boundaryIndices(9),[[0,1,2],[6,7,8],[0,3,6],[2,5,8]]);
        assert.deepEqual(display.boundaryIndices(4),[[0,1],[2,3],[0,2],[1,3]]);
        assert.throws(()=>display.boundaryIndices(8),/square/);
    });
    const context={console,THREE,...display,Float32Array};
    const source=fs.readFileSync(path.join(assets,'explorer.js'),'utf8').replace(/^import .*;\r?\n/gm,'').replace(/^export /gm,'');
    vm.createContext(context);vm.runInContext(source+'\nglobalThis.Explorer=Explorer;',context);
    check('Surface edges reuse full/preview buffers and obey the visibility switch',()=>{
        const e={scene:new THREE.Scene(),edgeCheck:{checked:true},is2D:false};
        const positions=new Float32Array(Array.from({length:27},(_,i)=>i));
        const update=(data,quality,count)=>context.Explorer.prototype.updateSurfaceEdges.call(e,data,quality,count);
        update(positions,'full',9);
        const objects=e.surfaceEdges.slice(),geometry=objects[0].geometry,material=objects[0].material;
        assert.equal(e.scene.children.length,4);
        update(new Float32Array(12),'preview',4);assert.notEqual(objects[0].geometry,geometry);
        positions[0]=7;update(positions,'full',9);
        assert.equal(objects[0].geometry,geometry);assert.equal(objects[0].material,material);
        assert.equal(geometry.attributes.instanceStart.getX(0),7);
        e.edgeCheck.checked=false;update(positions,'full',9);
        assert.ok(objects.every(line=>!line.visible));assert.equal(e.scene.children.length,4);
    });
    check('Reference layers provide a transparent grid without in-scene axis arrows',()=>{
        const box=new THREE.Box3(new THREE.Vector3(-2,-1,-3),new THREE.Vector3(4,5,2));
        for(const is2D of [true,false]){
            const layers=display.referenceLayers(is2D);layers.setBounds(box,10);
            assert.equal(layers.axes,undefined);assert.equal(layers.floor.material.transparent,true);
            assert.equal(layers.floor.material.depthWrite,false);
            assert.equal(layers.floor.position.x,1);assert.equal(layers.floor.position.y,2);
            if(is2D) assert.ok(layers.floor.position.z<0);
            else assert.ok(layers.floor.position.z<box.min.z);
        }
    });
    check('Planar tick positions use stable readable intervals across scales and negative ranges',()=>{
        assert.deepEqual(display.niceTicks(-2.4,3.1,6),[-2,-1,0,1,2,3]);
        assert.deepEqual(display.niceTicks(.001,.003,4),[.001,.0015,.002,.0025,.003]);
        assert.equal(display.tickLabel(.30000000000000004),'0.3');assert.equal(display.tickLabel(0),'0');
        assert.equal(display.tickLabel(200000),'2e+5');assert.deepEqual(display.niceTicks(1,1),[]);
        const frame=display.planarFrame(800,450);
        assert.equal(frame.left+frame.width+16,800);assert.equal(frame.top+frame.height+frame.bottom,450);
    });
    check('Ticked axes follow camera pan/zoom, cache unchanged frames, and release their overlay',()=>{
        class SVGNode {
            constructor(tag){this.tag=tag;this.attributes={};this.style={};this.children=[];}
            setAttribute(key,value){this.attributes[key]=value;}
            append(node){this.children.push(node);}
            replaceChildren(...nodes){this.children=nodes;}
            remove(){this.removed=true;}
        }
        global.document={createElementNS:(_,tag)=>new SVGNode(tag)};
        const parent=new SVGNode('div'),axes=display.createPlanarAxes(parent);
        const camera=new THREE.OrthographicCamera(-4,4,2,-2,.1,100);
        camera.position.z=10;camera.lookAt(0,0,0);camera.updateMatrixWorld(true);
        const update=()=>axes.update(camera,800,450,{enabled:true,grid:true});
        const xLabels=()=>axes.host.children.filter(node=>node.tag==='text' &&
            node.attributes['text-anchor']==='middle' && !node.attributes.class).map(node=>node.textContent);
        update();assert.ok(xLabels().includes('-4'));const first=axes.host.children;
        assert.ok(!axes.host.children.some(node=>node.attributes.class==='nrb-axis-grid'));
        update();assert.equal(axes.host.children,first);
        camera.position.x=2;camera.zoom=2;camera.updateProjectionMatrix();camera.updateMatrixWorld(true);
        update();assert.notEqual(axes.host.children,first);assert.ok(xLabels().includes('0'));
        assert.ok(xLabels().includes('4'));assert.ok(!xLabels().includes('-4'));
        axes.update(camera,800,450,{enabled:false,grid:false});assert.equal(axes.host.style.display,'none');
        axes.destroy();assert.equal(axes.host.removed,true);
    });
    check('Grid renders in a background pass with depth cleared before geometry',()=>{
        const events=[],background={},scene={},camera={};
        const renderer={setScissorTest(){},setViewport(){},setScissor(){},clear(){events.push('clear');},
            render(value){events.push(value===background?'grid':'geometry');},clearDepth(){events.push('depth');}};
        display.renderLayeredScene(renderer,background,scene,camera,800,450,display.planarFrame(800,450));
        assert.deepEqual(events,['clear','grid','depth','geometry']);
    });
    check('Planar handles retain a small pixel radius across zoom and viewport sizes',()=>{
        const camera=new THREE.OrthographicCamera(-4,4,2,-2,.1,100);
        for(const zoom of [1,2,5])for(const height of [250,600]){
            camera.zoom=zoom;
            const radius=display.planarHandleRadius(camera,height);
            assert.ok(Math.abs(radius/(camera.top-camera.bottom)*zoom*height-4.5)<1e-12);
        }
    });
    check('Spatial handles match planar outer size across camera distance, zoom, and viewport height',()=>{
        const camera=new THREE.PerspectiveCamera(38,1,.01,1000);
        camera.position.set(0,0,10);camera.lookAt(0,0,0);camera.updateMatrixWorld();
        for(const zoom of [.5,1,3])for(const height of [250,600])for(const z of [-5,0,5]){
            camera.zoom=zoom;camera.updateProjectionMatrix();
            const point=new THREE.Vector3(0,0,z),radius=display.spatialHandleRadius(camera,height,point);
            const center=point.clone().project(camera),edge=point.clone().add(new THREE.Vector3(0,radius,0)).project(camera);
            assert.ok(Math.abs((edge.y-center.y)*height/2-4.5*1.08)<1e-12);
        }
    });
    console.log(`${checks} display checks passed`);
})().catch(error=>{console.error(error);process.exitCode=1;});
