/* Figure state coordinates existing Explorer instances; all NURBS math stays in Python. */
import * as THREE from './vendor/three.module.js';
import {Explorer} from './explorer.js';
import {planarHandleRadius,spatialHandleRadius} from './display.js';
import {EditHistory} from './history.js';

const clone=value=>JSON.parse(JSON.stringify(value));
const mounted=new Map();
const frame=()=>new Promise(resolve=>requestAnimationFrame(resolve));

function displayState(owner){
    const s=owner.scientific;
    return {view:owner.viewMode,axes:owner.axesCheck.checked,grid:owner.gridCheck.checked,
        quantity:s.quantity.value,colormap:s.map.value,geometry_color:s.geometryColor.value,
        opacity:Number(s.opacity?.value??1),
        control_color:s.controlColor.value,selected_color:s.selectedColor?.value||'#2ca02c',
        mesh:s.meshMode?.value||'off',mesh_u:Number(s.nu?.value||40),mesh_v:Number(s.nv?.value||40),
        frames:Boolean(s.frames?.checked),frame_count:Number(s.frameCount?.value||8),frame_scale:Number(s.frameScale?.value||.1),
        reverse:s.reverse.checked,limits:s.auto.checked?null:[Number(s.minimum.value),Number(s.maximum.value)],
        boundary:owner.edgeCheck?.checked??true,control_polygon:owner.netCheck.checked,
        control_points:owner.handleCheck?.checked??true};
}

export async function mountFigure(rootId,payload,options={}){
    mounted.get(rootId)?.dispose();
    const root=document.getElementById(rootId);
    if(!root)throw new Error('Figure container is missing.');
    let state=clone(payload.state),owners=Object.create(null),core,selector,visibility,timer,disposed=false,lastPublished='';
    const originals=Object.fromEntries(Object.entries(payload.scenes).map(([name,scene])=>[name,clone(scene.model)]));
    let building=true;
    const selections=Object.create(null);
    const geometryBusy=()=>Object.values(owners).some(e=>e.pending||e.inFlight||e.drag||e.editing?.box||e.cameraAnimating);
    const history=options.readOnly?null:new EditHistory({
        busy:()=>building||geometryBusy(),
        changed:()=>{core?.editing?.updateHistory();publish();},
        restore:async(name,model,selection)=>{
            if(state.active!==name)await activate(name);
            await core.editing.restore(model,selection);
        },
    });
    const names=()=>Object.keys(state.objects);
    const busy=()=>geometryBusy()||Boolean(history?.restoring);
    const diagnosticsBusy=()=>Object.values(owners).some(e=>e.scientific.running||e.scientific.pending);
    function capture(ignoreHistory=false){
        if(ignoreHistory?geometryBusy():busy())throw new Error('Wait for the current edit or camera animation to finish.');
        const result=clone(state);
        for(const [name,e] of Object.entries(owners)){
            result.objects[name].display=displayState(e);
            if(!options.readOnly&&JSON.stringify(e.acceptedModel)!==JSON.stringify(payload.scenes[name].model)){
                const p=result.objects[name].geometry.parameters;
                p.control_points={dtype:'float64',values:clone(e.acceptedModel.control_points)};
                p.weights={dtype:'float64',values:clone(e.acceptedModel.weights)};
            }
        }
        const camera=core.camera;
        result.camera={projection:core.viewMode,navigation:core.is2D?(core.lastSpatialNavigation||'arcball'):core.navigation,
            position:camera.position.toArray(),target:core.controls.target.toArray(),
            up:new THREE.Vector3(0,1,0).applyQuaternion(camera.quaternion).toArray(),
            zoom:camera.zoom,fov:camera.fov||state.camera?.fov||38,near:camera.near,far:camera.far,
            ortho_half:core.orthoHalf||3};
        result.active=state.active;
        return result;
    }
    function publish(){
        if(building||disposed||!options.onState||busy())return;
        clearTimeout(timer);
        timer=setTimeout(()=>{
            if(disposed||busy())return;
            const snapshot=capture(),encoded=JSON.stringify(snapshot);
            if(encoded!==lastPublished){lastPublished=encoded;options.onState(snapshot);}
        },80);
    }
    const inspect=name=>async packet=>{
        if(options.inspect)return options.inspect(name,packet);
        return {...clone(payload.scenes[name].diagnostics),revision:packet.revision,quality:packet.quality};
    };
    function bounds(){
        const points=Object.entries(owners).filter(([name])=>state.objects[name].visible)
            .flatMap(([,owner])=>owner.handles.map(h=>h.position));
        for(const marker of state.markers||[])for(let i=0;i<marker.points[0].length;i++)points.push(new THREE.Vector3(...marker.points.map(axis=>axis[i])));
        for(const mesh of state.meshes||[])for(const vertex of mesh.vertices)points.push(new THREE.Vector3(...vertex));
        return new THREE.Box3().setFromPoints(points.length?points:[new THREE.Vector3()]);
    }
    function beforeRender(){
        for(const [name,e] of Object.entries(owners)){
            const visible=state.objects[name].visible;
            e.scientific.legend.style.display=e.scientific.frameLegend.style.display=visible?'':'none';
            if(e===core){
                if(e.geometryObject)e.geometryObject.visible=visible&&e.scientific.meshMode?.value!=='export';
                if(e.net)e.net.visible=visible&&e.netCheck.checked;
                e.handles.forEach(h=>h.visible=visible&&(e.handleCheck?.checked??true));
                e.surfaceEdges?.forEach(edge=>edge.visible=visible&&(e.edgeCheck?.checked??true));
                if(e.scientific.frameGroup)e.scientific.frameGroup.visible=visible;
                if(e.scientific.wire)e.scientific.wire.visible=visible&&e.scientific.meshMode?.value!=='off';
                if(e.scientific.exportFill)e.scientific.exportFill.visible=visible&&e.scientific.meshMode?.value==='export';
            }else e.scene.visible=visible;
            if(e===core)continue;
            if(e.is2D!==core.is2D){e.is2D=core.is2D;e.setHandleStyle();e.scientific.applyColors();e.updateSelection();}
            e.viewMode=core.viewMode;e.camera=core.camera;e.controls=core.controls;
            for(const handle of e.handles){
                handle.quaternion.copy(core.is2D?core.camera.quaternion:new THREE.Quaternion());
                const height=core.plotRectangle().height;
                handle.scale.setScalar(core.is2D?planarHandleRadius(core.camera,height):spatialHandleRadius(core.camera,height,handle.position));
            }
        }
    }
    function restoreCamera(saved){
        if(!saved){core.fit();return;}
        core.setViewMode(saved.projection);
        core.lastSpatialNavigation=saved.navigation;
        const camera=core.camera;
        camera.position.fromArray(saved.position);camera.up.fromArray(saved.up);
        camera.near=saved.near;camera.far=saved.far;camera.zoom=saved.zoom;
        if(!core.is2D)camera.fov=saved.fov;
        core.orthoHalf=saved.ortho_half;
        core.controls.target.fromArray(saved.target);camera.lookAt(core.controls.target);
        core.setNavigation(saved.navigation,{target:new THREE.Vector3(...saved.target),cancelCube:false});
        core.resize();core.render();
    }
    function clear(){
        for(const [name,e] of Object.entries(owners))if(e.editing)selections[name]=e.editing.selection();
        for(const [name,e] of Object.entries(owners)){
            if(e===core)continue;
            core.scene.remove(e.scene);e.removalObserver.disconnect();e.scientific.destroy();e.planarAxes.destroy();
            e.root.remove();e.disposed=true;
            const geometries=new Set(),materials=new Set();
            e.scene.traverse(o=>{if(o.geometry)geometries.add(o.geometry);if(o.material)materials.add(o.material);});
            [e.pointGeometry,e.outlineGeometry].filter(Boolean).forEach(g=>geometries.add(g));
            [e.pointMaterial,e.selectedMaterial,e.outlineMaterial,e.selectedOutlineMaterial].filter(Boolean).forEach(m=>materials.add(m));
            geometries.forEach(g=>g.dispose());materials.forEach(m=>m.dispose());
        }
        core?.dispose();owners=Object.create(null);
    }
    function build(saved){
        building=true;
        const active=state.active;
        const sceneFor=name=>({...clone(payload.scenes[name]),title:name===active?state.title:name,display:state.objects[name].display});
        core=new Explorer(root,sceneFor(active),{
            readOnly:Boolean(options.readOnly),aspect:state.rendering.width/state.rendering.height,
            inspect:inspect(active),evaluate:options.evaluate?packet=>options.evaluate(active,packet):undefined,
            beforeRender,afterRender:publish,onBusyChange:publish,
            history,historyKey:active,
        });
        owners[active]=core;
        core.original=clone(originals[active]||payload.scenes[active].model);
        if(selections[active])core.setSelection(selections[active].indices,selections[active].active);
        for(const name of names().filter(n=>n!==active)){
            const hidden=document.createElement('div');hidden.hidden=true;root.append(hidden);
            const owner=new Explorer(hidden,sceneFor(name),{readOnly:true,shared:core,inspect:inspect(name)});
            owners[name]=owner;core.scene.add(owner.scene);
            // Legends belong to the figure, not to the hidden object's inspector.
            core.viewport.append(owner.scientific.legend,owner.scientific.frameLegend);
        }
        core.options.bounds=bounds;
        for(const marker of state.markers||[]){
            const positions=new Float32Array(marker.points[0].length*3);
            for(let i=0;i<marker.points[0].length;i++)for(let d=0;d<3;d++)positions[3*i+d]=marker.points[d][i];
            const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));
            const material=new THREE.PointsMaterial({color:marker.color,size:marker.size,sizeAttenuation:false,
                depthTest:marker.depth_test,depthWrite:false,transparent:true});
            material.onBeforeCompile=shader=>{shader.fragmentShader=shader.fragmentShader.replace('void main() {',
                'void main() { if (length(gl_PointCoord-vec2(0.5))>0.5) discard;');};
            const points=new THREE.Points(geometry,material);points.renderOrder=40;points.frustumCulled=false;
            core.scene.add(points);
        }
        core.renderer.setClearColor(state.rendering.background,1);
        for(const record of state.meshes||[]){
            const geometry=new THREE.BufferGeometry();
            geometry.setAttribute('position',new THREE.Float32BufferAttribute(record.vertices.flat(),3));
            geometry.setIndex(record.triangles.flat());geometry.computeVertexNormals();
            const mesh=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color:record.color,side:THREE.DoubleSide,roughness:.65,metalness:.02}));
            core.scene.add(mesh);
            if(record.wireframe){
                const wire=new THREE.LineSegments(new THREE.WireframeGeometry(geometry),new THREE.LineBasicMaterial({color:'#4e6575',transparent:true,opacity:.3}));
                core.scene.add(wire);
            }
        }
        core.viewport.style.background=state.rendering.background;
        const tools=root.querySelector('.nrb-camera-tools');
        const label=document.createElement('label');label.className='nrb-navigation';
        label.hidden=names().length===1;
        label.append(document.createTextNode('Object'));
        selector=document.createElement('select');selector.setAttribute('aria-label','Figure object');
        for(const name of names()){const option=document.createElement('option');option.value=name;option.textContent=name;selector.append(option);}
        selector.value=active;label.append(selector);tools.prepend(label);
        const row=document.createElement('label');row.className='nrb-science-row';row.append(document.createTextNode('Object visible'));
        visibility=document.createElement('input');visibility.type='checkbox';visibility.setAttribute('aria-label','Object visible');
        visibility.checked=state.objects[active].visible;row.append(visibility);core.configSections.geometry.prepend(row);
        visibility.addEventListener('change',()=>{state.objects[active].visible=visibility.checked;core.render();});
        const backgroundRow=document.createElement('label');backgroundRow.className='nrb-science-row';backgroundRow.textContent='Background colour';
        const background=document.createElement('input');background.type='color';background.value=state.rendering.background;
        background.setAttribute('aria-label','Background colour');backgroundRow.append(background);core.configSections.view.append(backgroundRow);
        background.addEventListener('input',()=>{
            state.rendering.background=background.value;core.renderer.setClearColor(background.value,1);
            core.viewport.style.background=background.value;core.render();
        });
        selector.addEventListener('change',async()=>{
            const selected=selector.value;
            try{if(history?.locked){selector.value=state.active;return;}await activate(selected);}
            catch(error){selector.value=state.active;core.setStatus(error.message,true);}
        });
        // A shared selector changes inspectors; it never fits or moves the camera.
        const box=bounds(),extent=Math.max(box.getSize(new THREE.Vector3()).length(),1e-6);
        core.references?.setBounds(box,extent,core.is2D?core.viewMode:false);
        restoreCamera(saved);building=false;core.editing?.updateHistory();
        let legendIndex=0;
        for(const [name,e] of Object.entries(owners)){
            e.scientific.legend.style.top=`${17+legendIndex*84}px`;
            e.scientific.frameLegend.style.bottom=`${12+legendIndex*25}px`;
            if(state.objects[name].visible&&e.scientific.quantity.value!=='solid')legendIndex++;
        }
        if(options.export){
            core.renderer.setPixelRatio(window.devicePixelRatio||1);
            root.classList.add('nrb-figure-export');
            root.querySelector('.nrb-heading').hidden=true;
            root.querySelectorAll('.nrb-sidebar').forEach(e=>e.hidden=true);
            core.viewCube.host.hidden=true;
            root.style.border='0';root.style.borderRadius='0';root.style.minHeight='0';
            core.resize();
        }
    }
    async function activate(name){
        await ready(true);state=capture(true);
        for(const [key,e] of Object.entries(owners))payload.scenes[key].model=clone(e.acceptedModel);
        if(options.scenes)payload=await options.scenes(state);
        state.active=name;clear();build(state.camera);await ready(true);publish();
    }
    async function ready(ignoreHistory=false){
        for(let i=0;i<3600;i++){
            if(disposed)throw new Error('Figure has been disposed.');
            if(!(ignoreHistory?geometryBusy():busy())&&!diagnosticsBusy()){
                const error=Object.values(owners).find(e=>e.scientific.lastError)?.scientific.lastError;
                if(error)throw new Error(error);
                core.render();await frame();await frame();return;
            }
            await frame();
        }
        throw new Error('Figure did not finish rendering.');
    }
    async function updateFrame(next){
        if(!options.export)throw new Error('Batch frame updates require an export renderer.');
        await ready();
        const changed=Object.keys(next.scenes);
        if(next.patch)next.scenes=Object.fromEntries(Object.keys(next.state.objects).map(name=>[name,next.scenes[name]||payload.scenes[name]]));
        const reusable=state.active===next.state.active &&
            JSON.stringify(names())===JSON.stringify(Object.keys(next.state.objects)) &&
            JSON.stringify(state.markers||[])===JSON.stringify(next.state.markers||[]) &&
            JSON.stringify(state.meshes||[])===JSON.stringify(next.state.meshes||[]) &&
            names().every(name=>JSON.stringify(state.objects[name].display)===JSON.stringify(next.state.objects[name].display) &&
                JSON.stringify(payload.scenes[name].topologies)===JSON.stringify(next.scenes[name].topologies) &&
                JSON.stringify(payload.scenes[name].control_shape)===JSON.stringify(next.scenes[name].control_shape));
        const camera=next.state.camera||capture().camera;
        state=clone(next.state);payload=next;building=true;
        if(!reusable){clear();build(camera);}
        else{
            for(const name of changed){const e=owners[name];e.apply({...clone(payload.scenes[name]),revision:e.lastRevision+1});}
            core.renderer.setClearColor(state.rendering.background,1);core.viewport.style.background=state.rendering.background;
            restoreCamera(camera);building=false;
        }
        await ready();
    }
    const controller={snapshot:capture,ready,update:updateFrame,
        dispose(){disposed=true;clearTimeout(timer);clear();mounted.delete(rootId);},
    };
    build(state.camera);mounted.set(rootId,controller);await ready();publish();
    return controller;
}

export function figureState(rootId){
    const figure=mounted.get(rootId);if(!figure)throw new Error('Figure is not mounted.');
    return figure.snapshot();
}
