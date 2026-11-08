/* Scientific values come from Python. This module owns display controls only. */
import * as THREE from './vendor/three.module.js';
import {tickLabel,updateLineColors} from './display.js';

function node(tag,attributes={},text){
    const result=document.createElement(tag);
    for(const [key,value] of Object.entries(attributes)) result.setAttribute(key,value);
    if(text!==undefined) result.textContent=text;
    return result;
}

export function fieldLimits(field,manual){
    if(manual && manual.every(Number.isFinite) && manual[1]>manual[0]) return manual;
    if(!field.range) return null;
    let [low,high]=field.range;
    if(low===high){const delta=Math.max(Math.abs(low)*.01,1e-6);low-=delta;high+=delta;}
    return [low,high];
}

function applyOpacity(display){
    const opacity=Number(display.opacity?.value??1);
    for(const material of new Set([display.owner.geometryObject?.material,display.solidMaterial,display.colorMaterial,display.exportFill?.material])){
        if(!material)continue;
        material.opacity=opacity;
        if(opacity<1&&!material.transparent){material.transparent=true;material.needsUpdate=true;}
        if(display.isSurface)material.depthWrite=opacity>=1;
    }
}

export function fieldColors(values,table,limits,reverse=false){
    const result=new Float32Array(values.length*3),color=new THREE.Color();
    values.forEach((value,index)=>{
        if(value===null || !Number.isFinite(value)) color.set('#a6abb1');
        else{
            let fraction=Math.max(0,Math.min(1,(value-limits[0])/(limits[1]-limits[0])));
            if(reverse) fraction=1-fraction;
            const sample=fraction*(table.length-1),first=Math.floor(sample),next=Math.min(first+1,table.length-1);
            const rgb=table[first].map((x,i)=>x+(table[next][i]-x)*(sample-first));
            color.setRGB(...rgb,THREE.SRGBColorSpace);
        }
        color.toArray(result,index*3);
    });
    return result;
}

export class ScientificDisplay {
    constructor(owner,sections,initial){
        this.owner=owner;this.tables=initial.colormaps||{};this.isSurface=Boolean(initial.triangles?.length);
        this.serial=0;this.pending=null;this.running=false;this.alive=true;this.field=null;this.stl=null;
        const panel=owner.configPanel;let target=sections.geometry;
        const row=(label,control)=>{
            const labelNode=node('label',{class:'nrb-science-row'},label);labelNode.append(control);target.append(labelNode);return control;
        };
        const select=(label,values)=>{
            const control=node('select',{'aria-label':label});
            for(const [value,text] of values) control.append(node('option',{value},text));
            return row(label,control);
        };
        const number=(label,value,min,max,step=1)=>{
            const control=node('input',{type:'number','aria-label':label,min,max,step});control.value=value;return row(label,control);
        };
        const check=(label)=>{const control=node('input',{type:'checkbox','aria-label':label});return row(label,control);};
        this.geometryColor=row(this.isSurface?'Surface colour':'Curve colour',node('input',{type:'color','aria-label':'Geometry colour',value:this.isSurface?'#0072bd':'#000000'}));
        if(!this.isSurface){this.curveDepth=check('Hide behind surfaces');this.curveDepth.addEventListener('change',()=>this.applyColors());}
        this.opacity=number('Opacity',1,0,1,.05);
        this.opacity.addEventListener('input',()=>this.applyColors());
        target=sections.controls;
        this.controlColor=row('Points & polygon colour',node('input',{type:'color','aria-label':'Points and polygon colour',value:'#ff7f0e'}));
        if(!owner.readOnly)this.selectedColor=row('Selected point colour',node('input',{type:'color','aria-label':'Selected point colour',value:'#2ca02c'}));
        this.geometryColor.addEventListener('input',()=>this.applyColors());
        this.controlColor.addEventListener('input',()=>this.applyColors());
        this.selectedColor?.addEventListener('input',()=>this.applyColors());
        const restore=node('button',{type:'button',class:'nrb-colour-reset'},'Restore default colours');panel.append(restore);
        restore.addEventListener('click',()=>{
            this.geometryColor.value=this.isSurface?'#0072bd':'#000000';this.controlColor.value='#ff7f0e';if(this.selectedColor)this.selectedColor.value='#2ca02c';this.applyColors();
        });
        target=sections.scientific;
        this.quantity=select('Quantity',[['solid','Solid colour'],...Object.entries(initial.fields||{})]);
        this.map=select('Colour map',Object.keys(this.tables).map(name=>[name,name]));
        this.map.value='viridis';this.reverse=check('Reverse colour map');
        this.auto=check('Automatic limits');this.auto.checked=true;
        this.minimum=number('Minimum',0,-1e20,1e20,'any');this.maximum=number('Maximum',1,-1e20,1e20,'any');
        for(const input of [this.map,this.reverse,this.auto,this.minimum,this.maximum]) input.addEventListener('change',()=>{
            this.syncControls();
            if(!this.auto.checked && !(Number(this.maximum.value)>Number(this.minimum.value))){this.note.textContent='Maximum must exceed minimum.';return;}
            this.applyField();
        });
        this.quantity.addEventListener('change',()=>{this.field=null;this.heldLimits=null;this.lastLimits=null;this.applyColors();this.refresh();});
        this.minimum.disabled=this.maximum.disabled=true;
        target=sections.extras;
        if(this.isSurface){
            this.meshMode=select('Mesh',[['off','None'],['display','Display mesh'],['export','STL export mesh']]);
            this.meshHelp=node('small');target.append(this.meshHelp);
            this.nu=number('Samples along u',40,2,160);this.nv=number('Samples along v',40,2,160);
            this.download=node('button',{type:'button',disabled:''},'Download STL');target.append(this.download);
            this.download.addEventListener('click',()=>{
                if(!this.stl)return;
                const bytes=Uint8Array.from(atob(this.stl),character=>character.charCodeAt(0));
                const url=URL.createObjectURL(new Blob([bytes],{type:'model/stl'})),anchor=node('a',{href:url,download:'nurbspy_surface.stl'});
                document.body.append(anchor);anchor.click();anchor.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
            });
            for(const input of [this.meshMode,this.nu,this.nv]) input.addEventListener('change',()=>{
                this.quantity.disabled=this.meshMode.value==='export';this.field=null;this.refresh();
            });
        }else{
            this.frames=check('Show frames');this.frameCount=number('Number of frames',8,2,30);
            this.frameScale=number('Arrow scale',.1,.005,.5,.01);
            for(const input of [this.frames,this.frameCount,this.frameScale]) input.addEventListener('change',()=>this.refresh());
            target.append(node('small',{},'Frames use approximately equal arc-length spacing. Undefined frames are omitted.'));
        }
        this.note=node('small',{class:'nrb-science-note'});panel.append(this.note);
        this.legend=node('div',{class:'nrb-colorbar',hidden:''});
        this.legendTitle=node('div');this.gradient=node('div',{class:'nrb-colorbar-gradient'});
        const range=node('div',{class:'nrb-colorbar-range'});this.low=node('span');this.high=node('span');range.append(this.low,this.high);
        this.legend.append(this.legendTitle,this.gradient,range);owner.viewport.append(this.legend);
        this.frameLegend=node('div',{class:'nrb-frame-legend',hidden:''});owner.viewport.append(this.frameLegend);
        const settings=initial.display||{};
        this.opacity.value=settings.opacity??1;
        if(this.curveDepth)this.curveDepth.checked=settings.curve_depth_test!==false;
        for(const [key,input] of [['quantity',this.quantity],['colormap',this.map],['geometry_color',this.geometryColor],['control_color',this.controlColor],['selected_color',this.selectedColor],['mesh',this.meshMode],['mesh_u',this.nu],['mesh_v',this.nv],['frame_count',this.frameCount],['frame_scale',this.frameScale]]){
            if(input && settings[key]!==undefined)input.value=settings[key];
        }
        if(this.frames)this.frames.checked=Boolean(settings.frames);
        this.reverse.checked=Boolean(settings.reverse);
        this.auto.checked=!settings.limits;
        if(settings.limits){this.minimum.value=settings.limits[0];this.maximum.value=settings.limits[1];}
        owner.netCheck.checked=settings.control_polygon!==false;
        if(owner.handleCheck)owner.handleCheck.checked=settings.control_points!==false;
        if(owner.edgeCheck)owner.edgeCheck.checked=settings.boundary!==false;
        owner.axesCheck.checked=Boolean(settings.axes);owner.gridCheck.checked=settings.grid!==false;
        this.syncControls();
    }

    syncControls(){
        const exporting=this.meshMode?.value==='export',scalar=this.quantity.value!=='solid'&&!exporting;
        this.quantity.disabled=exporting;
        for(const control of [this.map,this.reverse,this.auto])control.parentElement.hidden=!scalar;
        for(const control of [this.minimum,this.maximum]){
            control.parentElement.hidden=!scalar||this.auto.checked;control.disabled=this.auto.checked;
        }
        if(this.download){
            for(const control of [this.nu,this.nv])control.parentElement.hidden=!exporting;
            this.download.hidden=!exporting;
            this.meshHelp.textContent=exporting?'Preview and download use the same triangles. Export sampling includes knot boundaries and removes collapsed triangles.':
                this.meshMode.value==='display'?'Shows the triangulation used for responsive rendering. Its resolution and diagonals can differ from the STL mesh.':
                'Display mesh shows the rendering triangles. STL export mesh previews the triangles that will be saved.';
        }
        if(this.frames)for(const control of [this.frameCount,this.frameScale])control.parentElement.hidden=!this.frames.checked;
    }

    applyColors(){
        const e=this.owner,control=this.controlColor.value,selected=this.selectedColor?.value||control;
        if(!this.isSurface&&e.geometryObject&&this.curveDepth){
            e.geometryObject.material.depthTest=this.curveDepth.checked;
            e.geometryObject.renderOrder=this.curveDepth.checked?0:10;
        }
        e.net?.material.color.set(control);
        if(e.is2D){e.outlineMaterial?.color.set(control);e.selectedOutlineMaterial?.color.set(selected);}
        else{e.pointMaterial?.color.set(control);e.selectedMaterial?.color.set(selected);e.selectedMaterial?.emissive.set(selected);}
        if(e.geometryObject){
            if(this.isSurface){
                if(!this.solidMaterial)this.solidMaterial=e.geometryObject.material;
                this.solidMaterial.color.set(this.geometryColor.value);
                if(!this.field || this.quantity.value==='solid') e.geometryObject.material=this.solidMaterial;
            }else if(!this.field || this.quantity.value==='solid'){
                e.geometryObject.material.vertexColors=false;e.geometryObject.material.color.set(this.geometryColor.value);
                e.geometryObject.material.needsUpdate=true;
            }
        }
        this.exportFill?.material.color.set(this.geometryColor.value);
        applyOpacity(this);
        if(this.quantity.value==='solid' || this.meshMode?.value==='export')this.legend.hidden=true;
        e.render();
    }

    geometryChanged(){
        this.stl=null;if(this.download)this.download.disabled=true;
        if(this.wire)this.wire.visible=false;if(this.exportFill)this.exportFill.visible=false;
        this.field=null;this.applyColors();this.refresh();
    }

    refresh(){
        if(!this.alive || !this.owner.geometryObject)return;
        this.lastError=null;
        this.syncControls();
        const e=this.owner,exporting=this.meshMode?.value==='export',preview=e.displayQuality==='preview';
        if(preview)this.heldLimits=this.heldLimits||this.lastLimits;
        else this.heldLimits=null;
        this.stl=null;if(this.download)this.download.disabled=true;
        if(this.wire)this.wire.visible=false;if(this.exportFill)this.exportFill.visible=false;
        e.geometryObject.visible=true;
        if(this.meshMode?.value==='display')this.updateWire(e.geometryObject.geometry.attributes.position.array,e.topologies[e.displayQuality].triangles);
        this.applyColors();
        const field=exporting?'solid':this.quantity.value;
        const frameEnabled=Boolean(this.frames?.checked);
        if(!frameEnabled)this.clearFrames();
        const exportEnabled=exporting&&!preview;
        const request={revision:++this.serial,quality:e.displayQuality||'full',
            control_points:(e.displayModel||e.acceptedModel).control_points,weights:(e.displayModel||e.acceptedModel).weights,
            field,frames:frameEnabled,frame_count:Number(this.frameCount?.value||8),frame_scale:Number(this.frameScale?.value||.1),
            export_mesh:exportEnabled,nu:Number(this.nu?.value||40),nv:Number(this.nv?.value||40)};
        if(field==='solid'&&!frameEnabled&&!exportEnabled){
            this.pending=null;this.legend.hidden=true;this.note.textContent=exporting?'Export mesh updates after releasing the point.':'';e.render();return;
        }
        if(!e.options.inspect){this.note.textContent='Scientific display requires the current Python adapter.';return;}
        this.pending=request;this.note.textContent=exportEnabled?'Generating export mesh and STL…':'Evaluating in Python…';this.flush();
    }

    async flush(){
        if(this.running || !this.pending || !this.alive)return;
        const request=this.pending;this.pending=null;this.running=true;
        let timeout;
        try{
            const result=await Promise.race([this.owner.options.inspect(request),new Promise((_,reject)=>{
                timeout=setTimeout(()=>reject(new Error('Python inspection timed out.')),30000);
            })]);
            if(!this.alive || request.revision!==this.serial)return;
            if(result.revision!==request.revision || result.quality!==request.quality)throw new Error('Unexpected inspection revision.');
            const notes=[];
            this.field=result.field||null;this.applyField();
            if(this.field?.undefined)notes.push(`${this.field.undefined} undefined values shown grey.`);
            if(result.frames){this.updateFrames(result.frames);if(result.frames.omitted)notes.push(`${result.frames.omitted} undefined frames omitted.`);}
            if(result.mesh){
                const mesh=result.mesh;this.updateWire(new Float32Array(mesh.vertices.flat()),mesh.triangles);
                this.updateExportFill(mesh);this.stl=mesh.stl;this.download.disabled=false;
                notes.push(`${mesh.triangles.length.toLocaleString()} export triangles. STL generated in memory. Export preview uses solid colour.`);
            }else if(this.meshMode?.value==='display')notes.push(`${this.owner.topologies[this.owner.displayQuality].triangles.length.toLocaleString()} display triangles.`);
            this.note.textContent=notes.join(' ');this.owner.render();
        }catch(error){
            this.lastError=error.message;
            if(this.alive&&request.revision===this.serial){this.note.textContent=error.message;this.field=null;this.legend.hidden=true;this.applyColors();}
        }finally{
            clearTimeout(timeout);this.running=false;if(this.pending)this.flush();
        }
    }

    applyField(){
        const e=this.owner,field=this.field;
        if(!field || this.quantity.value==='solid' || this.meshMode?.value==='export'){this.legend.hidden=true;this.applyColors();return;}
        const limits=fieldLimits(field,this.auto.checked?this.heldLimits:[Number(this.minimum.value),Number(this.maximum.value)]);
        if(!limits){this.legend.hidden=true;this.note.textContent='The selected quantity is undefined on this geometry.';return;}
        this.lastLimits=limits;
        const table=this.tables[this.map.value];if(!table)return;
        const rgb=fieldColors(field.values,table,limits,this.reverse.checked),geometry=e.geometryObject.geometry;
        if(this.isSurface){
            let attribute=geometry.attributes.color;
            if(!attribute || attribute.array.length!==rgb.length){attribute=new THREE.BufferAttribute(rgb,3);geometry.setAttribute('color',attribute);}
            else{attribute.array.set(rgb);attribute.needsUpdate=true;}
            if(!this.colorMaterial)this.colorMaterial=new THREE.MeshBasicMaterial({vertexColors:true,side:THREE.DoubleSide,polygonOffset:true,polygonOffsetFactor:1,polygonOffsetUnits:1});
            e.geometryObject.material=this.colorMaterial;
            applyOpacity(this);
        }else{
            updateLineColors(geometry,rgb);e.geometryObject.material.color.set('#ffffff');
            e.geometryObject.material.vertexColors=true;e.geometryObject.material.needsUpdate=true;
        }
        this.legend.hidden=false;this.legendTitle.textContent=field.label+(field.unit?` (${field.unit})`:'');
        const stops=(this.reverse.checked?[...table].reverse():table).map(rgb=>`rgb(${rgb.map(x=>Math.round(x*255)).join(',')})`);
        this.gradient.style.background=`linear-gradient(to right,${stops.join(',')})`;
        this.low.textContent=tickLabel(limits[0]);this.high.textContent=tickLabel(limits[1]);
        if(this.auto.checked){this.minimum.value=limits[0];this.maximum.value=limits[1];}
        e.render();
    }

    updateWire(vertices,triangles){
        const edges=new Set(),points=[];
        for(const face of triangles)for(let i=0;i<3;i++){
            const a=face[i],b=face[(i+1)%3],key=a<b?`${a},${b}`:`${b},${a}`;
            if(edges.has(key))continue;edges.add(key);
            for(const index of [a,b])points.push(vertices[index*3],vertices[index*3+1],vertices[index*3+2]);
        }
        if(!this.wire){
            this.wire=new THREE.LineSegments(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:'#334552',transparent:true,opacity:.6,depthWrite:false}));
            this.wire.renderOrder=3;this.wire.frustumCulled=false;this.owner.scene.add(this.wire);
        }
        const attribute=this.wire.geometry.attributes.position;
        if(attribute?.array.length===points.length){attribute.array.set(points);attribute.needsUpdate=true;}
        else{
            this.wire.geometry.dispose();this.wire.geometry=new THREE.BufferGeometry();
            this.wire.geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(points),3));
        }
        this.wire.visible=true;this.owner.render();
    }

    updateExportFill(mesh){
        if(!this.exportFill){
            this.exportFill=new THREE.Mesh(new THREE.BufferGeometry(),new THREE.MeshStandardMaterial({color:this.geometryColor.value,side:THREE.DoubleSide,roughness:.52,polygonOffset:true,polygonOffsetFactor:1,polygonOffsetUnits:1}));
            this.exportFill.frustumCulled=false;this.owner.scene.add(this.exportFill);
        }
        const positions=new Float32Array(mesh.vertices.flat()),indices=mesh.triangles.flat();
        let geometry=this.exportFill.geometry;
        if(geometry.attributes.position?.array.length!==positions.length || geometry.index?.array.length!==indices.length){
            geometry.dispose();geometry=new THREE.BufferGeometry();this.exportFill.geometry=geometry;
            geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));geometry.setIndex(indices);
        }else{
            geometry.attributes.position.array.set(positions);geometry.attributes.position.needsUpdate=true;
            geometry.index.array.set(indices);geometry.index.needsUpdate=true;
        }
        geometry.computeVertexNormals();
        this.exportFill.visible=true;this.owner.geometryObject.visible=false;this.legend.hidden=true;
        applyOpacity(this);
    }

    clearFrames(){
        if(this.frameGroup){
            this.frameGroup.traverse(object=>{object.material?.dispose();});
            this.owner.scene.remove(this.frameGroup);this.frameGroup=null;
        }
        this.frameLegend.hidden=true;
    }

    updateFrames(data){
        this.clearFrames();this.frameGroup=new THREE.Group();
        for(const frame of data.frames)for(const [name,color] of [['tangent','#c73d3d'],['normal','#2ca02c'],['binormal','#0072bd']]){
            const arrow=new THREE.ArrowHelper(new THREE.Vector3(...frame[name]),new THREE.Vector3(...frame.origin),data.length,color,data.length*.22,data.length*.1);
            // Own one geometry pair per widget, independent of Three.js globals.
            if(!this.frameLineGeometry){this.frameLineGeometry=arrow.line.geometry.clone();this.frameConeGeometry=arrow.cone.geometry.clone();}
            arrow.line.geometry=this.frameLineGeometry;arrow.cone.geometry=this.frameConeGeometry;
            this.frameGroup.add(arrow);
        }
        this.owner.scene.add(this.frameGroup);
        this.frameLegend.innerHTML='<span style="color:#c73d3d">T</span> · <span style="color:#2ca02c">N</span> · <span style="color:#0072bd">B</span>';
        this.frameLegend.hidden=!data.frames.length;
    }

    destroy(){
        this.alive=false;this.serial++;this.pending=null;this.clearFrames();
        this.frameLineGeometry?.dispose();this.frameConeGeometry?.dispose();
        this.colorMaterial?.dispose();this.solidMaterial?.dispose();this.legend.remove();this.frameLegend.remove();
    }
}
