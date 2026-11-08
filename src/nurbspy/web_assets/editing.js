/* Optional geometry editing capability. Rendering remains in the shared viewport. */
import * as THREE from './vendor/three.module.js';
import {createInspector} from './panels.js';
import {EditHistory} from './history.js';

function element(tag,attributes={},text){
    const node=document.createElement(tag);
    for(const [name,value] of Object.entries(attributes))node.setAttribute(name,value);
    if(text!==undefined)node.textContent=text;
    return node;
}

export class EditingControls {
    constructor(owner){
        this.owner=owner;this.panel=createInspector(owner,'edit','Edit','editing controls');
        this.history=owner.options?.history||new EditHistory({
            busy:()=>Boolean(owner.pending||owner.inFlight||owner.drag||owner.resampling||this.box),
            restore:(_,model,selection)=>this.restore(model,selection),changed:()=>this.updateHistory(),
        });
        this.historyKey=owner.options?.historyKey||owner.root.id;
        const group=title=>{
            const node=element('details',{class:'nrb-config-group',open:''});node.append(element('summary',{},title));
            const body=element('div',{class:'nrb-config-body'});node.append(body);this.panel.body.append(node);return body;
        };
        const row=(parent,label,control)=>{const field=element('label',{class:'nrb-science-row'},label);field.append(control);parent.append(field);return control;};
        const selection=group('Selection');
        const tools=element('div',{class:'nrb-selection-tools'});
        this.selectionCount=element('output',{'aria-live':'polite'},'1 selected');
        this.boxButton=element('button',{type:'button','aria-pressed':'false',title:'Select a rectangle, or Shift-drag in the viewport'},'Box select');
        this.boxButton.addEventListener('click',()=>{this.boxMode=!this.boxMode;this.boxButton.setAttribute('aria-pressed',String(this.boxMode));});
        tools.append(this.selectionCount,this.boxButton);selection.append(tools);
        this.pointSelect=row(selection,'Active control point',element('select',{'aria-label':'Selected control point',title:'Coordinates and weight below edit only this active point'}));
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
        this.weight=row(weights,'Active point weight',element('input',{type:'number',min:'.000001',max:'1000000',step:'.05','aria-label':'Control point weight'}));
        this.weightSlider=element('input',{type:'range',min:'-2',max:'2',step:'.01','aria-label':'Logarithmic weight slider',class:'nrb-edit-weight-slider'});weights.append(this.weightSlider);
        this.pointSelect.addEventListener('change',()=>owner.select(Number(this.pointSelect.value)));
        this.weightSlider.addEventListener('input',()=>this.editWeight(10**Number(this.weightSlider.value)));
        this.weightSlider.addEventListener('change',()=>this.commitChange());
        this.weight.addEventListener('change',()=>{
            const value=Number(this.weight.value);
            if(!Number.isFinite(value)||value<=0||value>1e6){owner.setStatus('Weight must be positive and at most 1,000,000.',true);this.updateSelection();return;}
            this.editWeight(value);this.commitChange();
        });
    }

    setPoints(points){
        this.pointSelect.replaceChildren(...points.map((_,index)=>element('option',{value:index},this.owner.pointLabel(index))));
    }

    updateSelection(){
        const e=this.owner;if(!e.model)return;
        if(this.selectionCount)this.selectionCount.textContent=`${e.selectedIndices?.size??1} selected`;
        this.updateHistory();
        if(!e.handles[e.selected]){this.pointSelect.value='';this.coordinates?.forEach(input=>input.value='');return;}
        this.pointSelect.value=String(e.selected);
        const weight=e.getWeight();this.weight.value=Number(weight.toPrecision(7));this.weightSlider.value=Math.log10(weight);
        this.coordinates?.forEach((input,axis)=>{
            if(document.activeElement!==input)input.value=Number(e.handles[e.selected].position.getComponent(axis).toPrecision(15));
        });
        this.refreshNormal();
    }

    selection(){return {indices:[...(this.owner.selectedIndices||[this.owner.selected])],active:this.owner.selected};}

    updateHistory(){
        const e=this.owner,history=this.history;
        if(e.undoButton)e.undoButton.disabled=!history?.canUndo;
        if(e.redoButton)e.redoButton.disabled=!history?.canRedo;
        const locked=Boolean(history?.restoring||history?.transaction?.committing||e.drag||e.resampling||this.box||
            ((e.pending||e.inFlight)&&!history?.transaction));
        if(this.pointSelect)this.pointSelect.disabled=locked;
        for(const input of [...(this.coordinates||[]),this.weight,this.weightSlider])
            if(input)input.disabled=locked||!e.handles?.[e.selected];
        if(e.resetButton)e.resetButton.disabled=Boolean(history?.locked||this.box);
    }

    beginChange(kind){
        const e=this.owner;if(!this.history)return true;
        const transaction=this.history.transaction;
        if(transaction)return transaction.object===this.historyKey&&!transaction.committing&&transaction.kind===kind;
        if(e.pending||e.inFlight||e.resampling)return false;
        if(!this.history.begin(this.historyKey,e.model,this.selection()))return false;
        this.history.transaction.kind=kind;return true;
    }

    commitChange(){
        const transaction=this.history?.transaction;
        if(this.history&&!transaction)return;
        if(transaction)transaction.committing=true;
        this.owner.enqueue('edit',true,'full');this.updateHistory();
    }

    applyModel(model,selection){
        const e=this.owner;e.model=JSON.parse(JSON.stringify(model));e.localGeneration++;
        const points=Array.from({length:e.handles.length},(_,index)=>[0,1,2].map(d=>e.shape.length===1?
            e.model.control_points[d][index]:e.model.control_points[d][Math.floor(index/e.shape[1])][index%e.shape[1]]));
        e.moveHandles(points);e.updateNet();this.normalKey=null;
        if(selection&&e.setSelection)e.setSelection(selection.indices,selection.active);
        else e.updateSelection();
        e.render();
    }

    restore(model,selection){
        return new Promise((resolve,reject)=>{
            this.restoreRequest={resolve,reject};this.applyModel(model,selection);
            this.owner.enqueue('edit',true,'full');
        });
    }

    acknowledge(data,flight){
        if(this.restoreRequest&&flight.quality==='full'){
            const request=this.restoreRequest;this.restoreRequest=null;
            if(data.error)request.reject(new Error(data.error));else request.resolve();
            return;
        }
        const transaction=this.history?.transaction;
        if(!transaction||flight.historyToken!==transaction.token)return;
        if(data.error){
            this.owner.pending=null;this.releaseDrag();
            this.history.cancel();this.applyModel(transaction.before,transaction.selection);
            if(JSON.stringify(transaction.before)!==JSON.stringify(this.owner.acceptedModel))this.owner.enqueue('edit',true,'full');
        }else if(flight.quality==='full'&&transaction.committing){this.history.finish(data.model);}
    }

    async travel(redo=false){
        try{return await this.history?.travel(redo);}
        catch(error){if(!this.owner.disposed)this.owner.setStatus(error.message,true);return false;}
    }

    keyDown(event){
        if(event.defaultPrevented)return false;
        if(event.key==='Escape'&&(this.box||this.owner.drag)){
            event.preventDefault();event.stopPropagation();this.cancelGesture();return true;
        }
        const tag=(event.target?.tagName||event.target?.tag||'').toLowerCase();
        if(['input','textarea','select'].includes(tag)||event.target?.isContentEditable||event.altKey)return false;
        if(!(event.ctrlKey||event.metaKey))return false;
        const key=event.key.toLowerCase();if(key!=='z'&&key!=='y')return false;
        event.preventDefault();event.stopPropagation();this.travel(key==='y'||event.shiftKey);return true;
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
        if(e.disposed||e.is2D||this.planeSelect.value!=='normal'||!e.handles?.[e.selected]||e.drag||e.inFlight||e.pending||this.normalRunning)return;
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
        this.setPositions([[index,position]]);
    }

    setPositions(positions){
        const e=this.owner;
        for(const [index,position] of positions){
            e.handles[index].position.copy(position);
            for(let axis=0;axis<3;axis++){
                if(e.shape.length===1)e.model.control_points[axis][index]=position.getComponent(axis);
                else e.model.control_points[axis][Math.floor(index/e.shape[1])][index%e.shape[1]]=position.getComponent(axis);
            }
        }
        e.localGeneration++;e.updateNet();e.updateSelection();e.render();
    }

    editCoordinate(axis,raw){
        const e=this.owner;if(e.readOnly||e.disposed||!e.handles[e.selected])return;
        const value=Number(raw);
        if(String(raw).trim()===''||!Number.isFinite(value)){
            e.setStatus('Coordinates must be finite numbers.',true);
            this.coordinates[axis].value=e.handles[e.selected].position.getComponent(axis);return;
        }
        const position=e.handles[e.selected].position.clone();if(position.getComponent(axis)===value)return;
        if(!this.beginChange('coordinate'))return;
        position.setComponent(axis,value);this.setPoint(e.selected,position);this.commitChange();
    }

    editWeight(value){
        const e=this.owner;if(e.readOnly||e.disposed||!e.handles[e.selected])return;
        if(!Number.isFinite(value)||value<=0||value>1e6||e.getWeight()===value||!this.beginChange('weight'))return;
        if(e.shape.length===1)e.model.weights[e.selected]=value;
        else e.model.weights[Math.floor(e.selected/e.shape[1])][e.selected%e.shape[1]]=value;
        e.localGeneration++;e.updateSelection();e.enqueue('edit',false,'preview');
    }

    reset(){
        const e=this.owner;if(e.readOnly||e.disposed)return;
        if(!this.beginChange('reset'))return;
        this.applyModel(e.original,{indices:[0],active:0});this.commitChange();
    }

    pointerDown(event){
        const e=this.owner;if(e.readOnly||e.disposed||event.button!==0)return;
        this.backgroundDown=null;
        const handle=e.pickControlPoint(event);
        e.renderer?.domElement?.focus?.({preventScroll:true});
        if(event.shiftKey||this.boxMode){this.startBox(event,handle);return;}
        if(!handle){this.backgroundDown={id:event.pointerId,x:event.clientX,y:event.clientY};return;}
        event.preventDefault();event.stopImmediatePropagation();e.select(handle.userData.index,{preserve:true});
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
        if(!this.beginChange('drag'))return;
        const starts=[...(e.selectedIndices||[e.selected])].map(index=>[index,e.handles[index].position.clone()]);
        e.viewCube?.cancel(false);
        e.drag={id:event.pointerId,index:e.selected,plane,axis,start:position,starts,hit,changed:false};e.controls.enabled=false;
        e.renderer.domElement.setPointerCapture(event.pointerId);e.renderer.domElement.style.cursor='grabbing';
        this.updateHistory();
    }

    startBox(event,handle){
        const e=this.owner;if(e.setRay(event)===false)return;
        event.preventDefault();event.stopImmediatePropagation();e.viewCube?.cancel(false);
        this.box={id:event.pointerId,x:event.clientX,y:event.clientY,lastX:event.clientX,lastY:event.clientY,
            indices:[...(e.selectedIndices||[e.selected])],additive:Boolean(event.shiftKey),index:handle?.userData.index};
        this.boxOverlay=element('div',{class:'nrb-selection-box',hidden:''});e.viewport.append(this.boxOverlay);
        e.controls.enabled=false;e.renderer.domElement.setPointerCapture(event.pointerId);this.updateHistory();
    }

    moveBox(event){
        const box=this.box,e=this.owner;if(event.pointerId!==box.id)return;
        event.preventDefault();event.stopImmediatePropagation();box.lastX=event.clientX;box.lastY=event.clientY;
        box.moved=Math.hypot(box.lastX-box.x,box.lastY-box.y)>=4;
        const bounds=e.viewport.getBoundingClientRect();this.boxOverlay.hidden=!box.moved;
        Object.assign(this.boxOverlay.style,{left:`${Math.min(box.x,box.lastX)-bounds.left}px`,
            top:`${Math.min(box.y,box.lastY)-bounds.top}px`,width:`${Math.abs(box.lastX-box.x)}px`,height:`${Math.abs(box.lastY-box.y)}px`});
    }

    finishBox(cancel=false){
        const e=this.owner,box=this.box;if(!box)return;
        this.box=null;this.boxOverlay?.remove();this.boxOverlay=null;
        if(!cancel){
            if(box.moved){
                const bounds=e.renderer.domElement.getBoundingClientRect(),rect=e.plotRectangle(bounds.width,bounds.height);
                e.camera.updateMatrixWorld();
                const indices=e.handles.flatMap((handle,index)=>{
                    if(handle.visible===false)return [];
                    const p=handle.position.clone().project(e.camera);
                    const x=bounds.left+rect.left+(p.x+1)*rect.width/2,y=bounds.top+rect.top+(1-p.y)*rect.height/2;
                    return p.z>=-1&&p.z<=1&&p.x>=-1&&p.x<=1&&p.y>=-1&&p.y<=1&&
                        x>=Math.min(box.x,box.lastX)&&x<=Math.max(box.x,box.lastX)&&
                        y>=Math.min(box.y,box.lastY)&&y<=Math.max(box.y,box.lastY)?[index]:[];
                });
                e.setSelection([...new Set([...(box.additive?box.indices:[]),...indices])],indices.at(-1));
            }else if(box.index!==undefined)e.select(box.index,{toggle:box.additive});
            else if(!box.additive)e.setSelection([]);
        }
        this.boxMode=false;this.boxButton?.setAttribute('aria-pressed','false');
        e.controls.enabled=true;if(e.renderer.domElement.hasPointerCapture(box.id))e.renderer.domElement.releasePointerCapture(box.id);
        this.updateHistory();
    }

    pointerMove(event){
        const e=this.owner;if(e.readOnly||e.disposed)return;
        if(this.box){this.moveBox(event);return;}
        const inside=e.setRay(event);
        if(!e.drag){e.renderer.domElement.style.cursor=inside!==false&&e.raycaster.intersectObjects(e.handles,false).length?'grab':'default';return;}
        if(event.pointerId!==e.drag.id)return;
        event.preventDefault();event.stopImmediatePropagation();
        const hit=e.raycaster.ray.intersectPlane(e.drag.plane,new THREE.Vector3());if(!hit)return;
        const delta=hit.sub(e.drag.hit);if(e.drag.axis){const distance=delta.dot(e.drag.axis);delta.copy(e.drag.axis).multiplyScalar(distance);}
        const positions=(e.drag.starts||[[e.drag.index,e.drag.start]]).map(([index,start])=>[index,start.clone().add(delta)]);
        if(positions.every(([index,position])=>position.equals(e.handles[index].position)))return;
        e.drag.changed=true;this.setPositions(positions);e.enqueue('edit',false,'preview');
    }

    releaseDrag(){
        const e=this.owner;if(!e.drag)return;
        const id=e.drag.id;e.drag=null;e.controls.enabled=true;
        if(e.renderer.domElement.hasPointerCapture(id))e.renderer.domElement.releasePointerCapture(id);
        e.renderer.domElement.style.cursor='grab';
    }

    endDrag(){
        if(!this.owner.drag)return;
        const changed=this.owner.drag?.changed;this.releaseDrag();
        if(changed)this.commitChange();else this.history?.cancel();
        this.updateHistory();
    }

    cancelGesture(){
        if(this.box){this.finishBox(true);return;}
        const transaction=this.history?.transaction;if(!this.owner.drag)return;
        this.releaseDrag();this.history?.cancel();
        if(transaction){this.applyModel(transaction.before,transaction.selection);this.owner.enqueue('edit',true,'full');}
    }

    pointerUp(event){
        const e=this.owner;
        const cancel=event.type==='pointercancel'||event.type==='lostpointercapture';
        if(this.box){if(event.pointerId===this.box.id){event.preventDefault();event.stopImmediatePropagation();this.finishBox(cancel);}return;}
        if(!e.drag){
            const down=this.backgroundDown;this.backgroundDown=null;
            if(down&&!cancel&&event.pointerId===down.id&&Math.hypot(event.clientX-down.x,event.clientY-down.y)<4)e.setSelection([]);
            return;
        }
        if(event.pointerId!==e.drag.id)return;
        event.preventDefault();event.stopImmediatePropagation();if(cancel)this.cancelGesture();else this.endDrag();
    }

    destroy(){
        this.finishBox(true);this.releaseDrag();
        if(this.restoreRequest){this.restoreRequest.reject(new Error('Editor was closed.'));this.restoreRequest=null;}
    }
}
