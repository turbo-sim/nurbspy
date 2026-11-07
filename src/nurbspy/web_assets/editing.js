/* Optional geometry editing capability. Rendering remains in the shared viewport. */
import * as THREE from './vendor/three.module.js';
import {createInspector} from './panels.js';

function element(tag,attributes={},text){
    const node=document.createElement(tag);
    for(const [name,value] of Object.entries(attributes))node.setAttribute(name,value);
    if(text!==undefined)node.textContent=text;
    return node;
}

export class EditingControls {
    constructor(owner){
        this.owner=owner;this.panel=createInspector(owner,'edit','Edit','editing controls');
        const group=title=>{
            const node=element('details',{class:'nrb-config-group',open:''});node.append(element('summary',{},title));
            const body=element('div',{class:'nrb-config-body'});node.append(body);this.panel.body.append(node);return body;
        };
        const row=(parent,label,control)=>{const field=element('label',{class:'nrb-science-row'},label);field.append(control);parent.append(field);return control;};
        const selection=group('Selection');
        this.pointSelect=row(selection,'Control point',element('select',{'aria-label':'Selected control point'}));
        this.coordinates=['X','Y','Z'].map((label,axis)=>{
            const input=row(selection,`${label} coordinate`,element('input',{type:'number',step:'any',class:'nrb-coordinate','aria-label':`${label} coordinate`}));
            input.addEventListener('change',()=>this.editCoordinate(axis,input.value));
            input.addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();this.editCoordinate(axis,input.value);}});
            return input;
        });
        const movement=group('Movement');
        this.planeSelect=row(movement,'Drag constraint',element('select',{'aria-label':'Drag constraint'}));
        for(const [value,label] of [['screen','View plane'],['xy','XY plane'],['xz','XZ plane'],['yz','YZ plane'],['x','X axis'],['y','Y axis'],['z','Z axis'],['normal','Normal direction']])this.planeSelect.append(element('option',{value},label));
        this.planeSelect.addEventListener('change',()=>{this.normalKey=null;this.refreshNormal();});
        const weights=group('Rational weight');
        this.weight=row(weights,'Weight',element('input',{type:'number',min:'.000001',max:'1000000',step:'.05','aria-label':'Control point weight'}));
        this.weightSlider=element('input',{type:'range',min:'-2',max:'2',step:'.01','aria-label':'Logarithmic weight slider',class:'nrb-edit-weight-slider'});weights.append(this.weightSlider);
        this.pointSelect.addEventListener('change',()=>owner.select(Number(this.pointSelect.value)));
        this.weightSlider.addEventListener('input',()=>this.editWeight(10**Number(this.weightSlider.value)));
        this.weightSlider.addEventListener('change',()=>owner.enqueue('edit',true,'full'));
        this.weight.addEventListener('change',()=>{
            const value=Number(this.weight.value);
            if(!Number.isFinite(value)||value<=0||value>1e6){owner.setStatus('Weight must be positive and at most 1,000,000.',true);this.updateSelection();return;}
            this.editWeight(value);owner.enqueue('edit',true,'full');
        });
    }

    setPoints(points){
        this.pointSelect.replaceChildren(...points.map((_,index)=>element('option',{value:index},this.owner.pointLabel(index))));
    }

    updateSelection(){
        const e=this.owner;if(!e.model||!e.handles[e.selected])return;
        this.pointSelect.value=String(e.selected);
        const weight=e.getWeight();this.weight.value=Number(weight.toPrecision(7));this.weightSlider.value=Math.log10(weight);
        this.coordinates?.forEach((input,axis)=>{
            if(document.activeElement!==input)input.value=Number(e.handles[e.selected].position.getComponent(axis).toPrecision(15));
        });
        this.refreshNormal();
    }

    syncProjection(){
        const e=this.owner;this.planeSelect.disabled=e.is2D;
        if(e.is2D){
            if(!this.projectionLocked)this.spatialConstraint=this.planeSelect.value;
            this.planeSelect.value=e.viewMode==='zx'?'xz':e.viewMode;this.projectionLocked=true;
        }else if(this.projectionLocked){
            this.planeSelect.value=this.spatialConstraint||'screen';this.projectionLocked=false;
        }
        this.refreshNormal();
    }

    updateNormalStatus(){
        const e=this.owner;
        if(!e.is2D&&this.planeSelect.value==='normal'&&this.normalReason)
            e.setStatus(this.normalDirection?'Normal direction ready.':this.normalReason,!this.normalDirection&&!this.normalRunning);
    }

    normalSignature(){return `${this.owner.selected}:${this.owner.localGeneration||0}`;}

    async refreshNormal(){
        const e=this.owner;
        if(e.disposed||e.is2D||this.planeSelect.value!=='normal'||e.drag||e.inFlight||e.pending||this.normalRunning)return;
        if(!e.options?.inspect){this.normalReason='The Python inspection connection is unavailable.';this.updateNormalStatus();return;}
        const key=this.normalSignature();if(this.normalKey===key)return;
        this.normalKey=key;this.normalDirection=null;this.normalReason='Evaluating normal direction…';this.normalRunning=true;this.updateNormalStatus();
        let timeout;
        try{
            const request={revision:e.localGeneration||0,control_points:JSON.parse(JSON.stringify(e.model.control_points)),
                weights:JSON.parse(JSON.stringify(e.model.weights)),normal_index:e.selected};
            const result=await Promise.race([e.options.inspect(request),new Promise((_,reject)=>{timeout=setTimeout(()=>reject(new Error('Normal evaluation timed out. Select the constraint again to retry.')),10000);})]);
            if(e.disposed||key!==this.normalSignature())return;
            if(result.error||result.revision!==request.revision)throw new Error(result.error||'The normal response did not match the current geometry.');
            const direction=result.normal?.direction;
            this.normalDirection=Array.isArray(direction)&&direction.length===3&&direction.every(Number.isFinite)&&Math.hypot(...direction)>1e-12?direction:null;
            this.normalReason=this.normalDirection?'Ready.':result.normal?.reason||'The geometric normal is undefined here.';
        }catch(error){if(!e.disposed&&key===this.normalSignature())this.normalReason=error.message;}
        finally{
            clearTimeout(timeout);this.normalRunning=false;
            if(!e.disposed){this.updateNormalStatus();if(key!==this.normalSignature())this.refreshNormal();}
        }
    }

    setPoint(index,position){
        const e=this.owner;e.handles[index].position.copy(position);
        for(let axis=0;axis<3;axis++){
            if(e.shape.length===1)e.model.control_points[axis][index]=position.getComponent(axis);
            else e.model.control_points[axis][Math.floor(index/e.shape[1])][index%e.shape[1]]=position.getComponent(axis);
        }
        e.localGeneration++;e.updateNet();e.updateSelection();e.render();
    }

    editCoordinate(axis,raw){
        const e=this.owner;if(e.readOnly||e.disposed)return;
        const value=Number(raw);
        if(String(raw).trim()===''||!Number.isFinite(value)){
            e.setStatus('Coordinates must be finite numbers.',true);
            this.coordinates[axis].value=e.handles[e.selected].position.getComponent(axis);return;
        }
        const position=e.handles[e.selected].position.clone();if(position.getComponent(axis)===value)return;
        position.setComponent(axis,value);this.setPoint(e.selected,position);e.enqueue('edit',true,'full');
    }

    editWeight(value){
        const e=this.owner;if(e.readOnly||e.disposed)return;
        if(e.shape.length===1)e.model.weights[e.selected]=value;
        else e.model.weights[Math.floor(e.selected/e.shape[1])][e.selected%e.shape[1]]=value;
        e.localGeneration++;e.updateSelection();e.enqueue('edit',false,'preview');
    }

    reset(){
        const e=this.owner;if(e.readOnly||e.disposed)return;
        if(e.drag)this.endDrag();e.model=JSON.parse(JSON.stringify(e.original));e.localGeneration++;
        const points=Array.from({length:e.handles.length},(_,index)=>[0,1,2].map(d=>e.shape.length===1?e.model.control_points[d][index]:e.model.control_points[d][Math.floor(index/e.shape[1])][index%e.shape[1]]));
        e.moveHandles(points);e.updateNet();e.select(0);e.enqueue('reset',true);
    }

    pointerDown(event){
        const e=this.owner;if(e.readOnly||e.disposed||event.button!==0)return;
        const handle=e.pickControlPoint(event);if(!handle)return;
        event.preventDefault();event.stopImmediatePropagation();e.select(handle.userData.index);
        const position=handle.position.clone(),mode=e.is2D?(e.viewMode==='zx'?'xz':e.viewMode):this.planeSelect.value;
        const axes={x:new THREE.Vector3(1,0,0),y:new THREE.Vector3(0,1,0),z:new THREE.Vector3(0,0,1)};
        let axis=null,normal;
        if(mode==='normal'){
            if(this.normalKey!==this.normalSignature()||!this.normalDirection){
                this.refreshNormal();e.setStatus(this.normalReason||'Select the point and wait for its normal before dragging.',!this.normalRunning);return;
            }
            axis=new THREE.Vector3(...this.normalDirection);
        }else if(mode.length===1)axis=axes[mode];
        if(axis){
            normal=e.camera.getWorldDirection(new THREE.Vector3());normal.addScaledVector(axis,-normal.dot(axis));
            if(normal.lengthSq()<1e-8){e.setStatus('Rotate the view to drag along this axis.');return;}normal.normalize();
        }else normal=mode==='xy'?axes.z:mode==='xz'?axes.y:mode==='yz'?axes.x:e.camera.getWorldDirection(new THREE.Vector3());
        const plane=new THREE.Plane().setFromNormalAndCoplanarPoint(normal,position);
        const hit=e.raycaster.ray.intersectPlane(plane,new THREE.Vector3());if(!hit)return;
        e.drag={id:event.pointerId,index:e.selected,plane,axis,start:position,hit};e.controls.enabled=false;
        e.renderer.domElement.setPointerCapture(event.pointerId);e.renderer.domElement.style.cursor='grabbing';
    }

    pointerMove(event){
        const e=this.owner;if(e.readOnly||e.disposed)return;
        const inside=e.setRay(event);
        if(!e.drag){e.renderer.domElement.style.cursor=inside!==false&&e.raycaster.intersectObjects(e.handles,false).length?'grab':'default';return;}
        if(event.pointerId!==e.drag.id)return;
        event.preventDefault();event.stopImmediatePropagation();
        const hit=e.raycaster.ray.intersectPlane(e.drag.plane,new THREE.Vector3());if(!hit)return;
        const delta=hit.sub(e.drag.hit);if(e.drag.axis){const distance=delta.dot(e.drag.axis);delta.copy(e.drag.axis).multiplyScalar(distance);}
        this.setPoint(e.drag.index,e.drag.start.clone().add(delta));e.enqueue('edit',false,'preview');
    }

    endDrag(){
        const e=this.owner;if(!e.drag)return;
        const id=e.drag.id;e.drag=null;e.controls.enabled=true;
        if(e.renderer.domElement.hasPointerCapture(id))e.renderer.domElement.releasePointerCapture(id);
        e.renderer.domElement.style.cursor='grab';e.enqueue('edit',true,'full');
    }

    pointerUp(event){
        const e=this.owner;if(!e.drag||event.pointerId!==e.drag.id)return;
        event.preventDefault();event.stopImmediatePropagation();this.endDrag();
    }
}
