/* Display primitives only. NURBS evaluation remains in native Python. */
import * as THREE from './vendor/three.module.js';
import {Line2} from './vendor/Line2.js';
import {LineGeometry} from './vendor/LineGeometry.js';
import {LineMaterial} from './vendor/LineMaterial.js';

export function lineGeometry(positions) {
    return new LineGeometry().setPositions(positions);
}

export function updateLinePositions(geometry, positions) {
    // Reuse the interleaved instance buffer; LineGeometry.setPositions() would
    // allocate new GPU attributes on every Python response.
    const buffer=geometry.attributes.instanceStart.data, count=positions.length/3-1;
    if (buffer.array.length!==count*6) throw new Error('Unexpected line sample count.');
    for(let i=0;i<count;i++) {
        buffer.array.set(positions.subarray(i*3,i*3+3),i*6);
        buffer.array.set(positions.subarray(i*3+3,i*3+6),i*6+3);
    }
    buffer.needsUpdate=true;
}

export function updateLineColors(geometry, colors) {
    const attribute=geometry.attributes.instanceColorStart;
    if(!attribute){geometry.setColors(colors);return;}
    const buffer=attribute.data,count=colors.length/3-1;
    if(buffer.array.length!==count*6)throw new Error('Unexpected line colour count.');
    for(let i=0;i<count;i++){
        buffer.array.set(colors.subarray(i*3,i*3+3),i*6);
        buffer.array.set(colors.subarray(i*3+3,i*3+6),i*6+3);
    }
    buffer.needsUpdate=true;
}

export function wideLine(geometry,{color='#000000',width=2.6,depthTest=true,order=0}={}) {
    const line=new Line2(geometry,new LineMaterial({color,linewidth:width,
        worldUnits:false,alphaToCoverage:true,transparent:true,opacity:1,
        depthTest,depthWrite:false,polygonOffset:true,polygonOffsetFactor:-1,polygonOffsetUnits:-1}));
    line.renderOrder=order;line.frustumCulled=false;
    return line;
}

export function boundaryIndices(sampleCount) {
    const side=Math.round(Math.sqrt(sampleCount));
    if(side<2 || side*side!==sampleCount) throw new Error('Expected a square surface sampling grid.');
    const indices=Array.from({length:side},(_,i)=>i);
    // These reference the native Python sampling grid; no curve is evaluated here.
    return [indices,indices.map(i=>(side-1)*side+i),
            indices.map(i=>i*side),indices.map(i=>i*side+side-1)];
}

export function referenceLayers(initialPlanar=false) {
    const floor=new THREE.Mesh(new THREE.PlaneGeometry(1,1),new THREE.ShaderMaterial({
        transparent:true,depthWrite:false,side:THREE.DoubleSide,
        uniforms:{gridColor:{value:new THREE.Color('#8795a2')},steps:{value:24}},
        vertexShader:'varying vec2 planeUV; void main(){planeUV=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}',
        fragmentShader:`varying vec2 planeUV; uniform vec3 gridColor; uniform float steps;
            void main(){
                vec2 p=(planeUV-.5)*steps;
                vec2 pixel=max(fwidth(p),vec2(.0001));
                vec2 d=abs(fract(p-.5)-.5)/pixel;
                float line=1.0-min(min(d.x,d.y),1.0);
                float fade=1.0-smoothstep(.20,.56,length(planeUV-.5));
                gl_FragColor=vec4(gridColor,.42*line*fade);
                #include <tonemapping_fragment>
                #include <colorspace_fragment>
            }`,
    }));
    floor.renderOrder=-10;floor.frustumCulled=false;
    const tickGrid=new THREE.LineSegments(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:'#c9d0d6',transparent:true,opacity:.65,depthWrite:false,depthTest:false}));
    tickGrid.frustumCulled=false;tickGrid.visible=false;
    let tickKey='';
    return {floor,tickGrid,updateTicks(camera,width,height,{enabled,plane='xy'}){
        tickGrid.visible=enabled;if(!enabled)return;
        const bounds=planarBounds(camera,width,height,plane);
        const key=[width,height,plane,...bounds.low.toArray(),...bounds.high.toArray()].join(',');
        if(key===tickKey)return;tickKey=key;
        const {axes,xLow,xHigh,yLow,yHigh,xTicks,yTicks}=bounds,positions=[];
        const point=(x,y)=>{const value=bounds.low.clone();value.setComponent(axes.horizontal,x);value.setComponent(axes.vertical,y);positions.push(...value.toArray());};
        for(const x of xTicks){point(x,yLow);point(x,yHigh);}
        for(const y of yTicks){point(xLow,y);point(xHigh,y);}
        const attribute=tickGrid.geometry.attributes.position;
        if(attribute?.array.length===positions.length){attribute.array.set(positions);attribute.needsUpdate=true;}
        else{tickGrid.geometry.dispose();tickGrid.geometry=new THREE.BufferGeometry();tickGrid.geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(positions),3));}
    },setBounds(box,extent,planar=initialPlanar){
        const center=box.getCenter(new THREE.Vector3());
        floor.scale.set(extent*2.7,extent*2.7,1);
        floor.position.copy(center);
        const normal=typeof planar==='string'?projectionPlanes[planar].normal:2;
        floor.position.setComponent(normal,box.min.getComponent(normal)-(planar?.004:.12)*extent);
    }};
}

export function planarFrame(width,height) {
    const left=width<320?46:54, right=16, top=16, bottom=44;
    return {left,top,bottom,width:Math.max(1,width-left-right),height:Math.max(1,height-top-bottom)};
}

export function niceTicks(minimum,maximum,target=6) {
    const span=maximum-minimum;
    if(!Number.isFinite(span) || span<=0) return [];
    const raw=span/target, power=10**Math.floor(Math.log10(raw));
    const step=[1,2,2.5,5,10].find(value=>value*power>=raw)*power;
    const first=Math.ceil((minimum-step*1e-9)/step), last=Math.floor((maximum+step*1e-9)/step);
    return Array.from({length:Math.max(0,Math.min(100,last-first+1))},(_,i)=>{
        const value=(first+i)*step;
        return Math.abs(value)<step*1e-10?0:Number(value.toPrecision(12));
    });
}

export function tickLabel(value) {
    if(value===0) return '0';
    if(Math.abs(value)>=1e5 || Math.abs(value)<1e-3) return value.toExponential(2).replace(/\.00e/,'e');
    return String(Number(value.toPrecision(7)));
}

export const projectionPlanes=Object.freeze({
    xy:{horizontal:0,vertical:1,normal:2,labels:['x','y'],up:[0,1,0],direction:[0,0,1]},
    yz:{horizontal:1,vertical:2,normal:0,labels:['y','z'],up:[0,0,1],direction:[1,0,0]},
    zx:{horizontal:2,vertical:0,normal:1,labels:['z','x'],up:[1,0,0],direction:[0,1,0]},
});

export function planarBounds(camera,width,height,plane='xy') {
    const frame=planarFrame(width,height),axes=projectionPlanes[plane];
    const low=new THREE.Vector3(-1,-1,0).unproject(camera),high=new THREE.Vector3(1,1,0).unproject(camera);
    const xLow=low.getComponent(axes.horizontal),xHigh=high.getComponent(axes.horizontal);
    const yLow=low.getComponent(axes.vertical),yHigh=high.getComponent(axes.vertical);
    return {frame,axes,low,high,xLow,xHigh,yLow,yHigh,
        xTicks:niceTicks(xLow,xHigh,Math.max(3,frame.width/85)),
        yTicks:niceTicks(yLow,yHigh,Math.max(3,frame.height/65))};
}

export function planarHandleRadius(camera,height,pixels=4.5) {
    return (camera.top-camera.bottom)/camera.zoom*pixels/Math.max(height,1);
}

export function spatialHandleRadius(camera,height,position,pixels=4.5) {
    // Match the planar circle's outer ring (radius 1.08), at every depth/zoom.
    const m=camera.matrixWorldInverse.elements;
    const depth=-(m[2]*position.x+m[6]*position.y+m[10]*position.z+m[14]);
    return 2*Math.max(depth,camera.near)*pixels*1.08/(camera.projectionMatrix.elements[5]*Math.max(height,1));
}

export function renderLayeredScene(renderer,background,scene,camera,width,height,rectangle=null) {
    renderer.setScissorTest(false);renderer.setViewport(0,0,width,height);renderer.clear();
    if(rectangle){
        renderer.setViewport(rectangle.left,rectangle.bottom,rectangle.width,rectangle.height);
        renderer.setScissor(rectangle.left,rectangle.bottom,rectangle.width,rectangle.height);renderer.setScissorTest(true);
    }
    // Grids never compete with geometry for depth or transparent draw ordering.
    renderer.render(background,camera);renderer.clearDepth();renderer.render(scene,camera);
    renderer.setScissorTest(false);renderer.setViewport(0,0,width,height);
}

export function createPlanarAxes(parent) {
    const svgNS='http://www.w3.org/2000/svg';
    const make=(tag,attributes={},text)=>{
        const node=document.createElementNS(svgNS,tag);
        for(const [name,value] of Object.entries(attributes)) node.setAttribute(name,value);
        if(text!==undefined) node.textContent=text;
        return node;
    };
    const host=make('svg',{'class':'nrb-planar-axes','aria-hidden':'true'});
    host.hidden=true;parent.append(host);
    let previous='';
    return {host,update(camera,width,height,{enabled,grid,plane='xy'}){
        host.style.display=enabled?'block':'none';
        if(!enabled) return;
        const {frame,axes,xLow,xHigh,yLow,yHigh,xTicks,yTicks}=planarBounds(camera,width,height,plane);
        const key=[width,height,xLow,yLow,xHigh,yHigh,grid,plane].join(',');
        if(key===previous) return;
        previous=key;host.hidden=false;host.setAttribute('viewBox',`0 0 ${width} ${height}`);
        const {left,top}=frame, right=left+frame.width,bottom=top+frame.height;
        const xPixel=value=>left+(value-xLow)/(xHigh-xLow)*frame.width;
        const yPixel=value=>bottom-(value-yLow)/(yHigh-yLow)*frame.height;
        const children=[];
        const text=(x,y,label,anchor='middle',extra={})=>make('text',{x,y,'text-anchor':anchor,...extra},label);
        for(const value of xTicks){
            const x=xPixel(value);
            children.push(make('path',{d:`M${x},${bottom}v5 M${x},${top}v4`}),text(x,bottom+19,tickLabel(value)));
        }
        for(const value of yTicks){
            const y=yPixel(value);
            children.push(make('path',{d:`M${left},${y}h-5 M${right},${y}h-4`}),text(left-9,y+3.5,tickLabel(value),'end'));
        }
        children.push(make('rect',{x:left,y:top,width:frame.width,height:frame.height,fill:'none'}));
        children.push(text(left+frame.width/2,height-7,axes.labels[0],'middle',{'class':'nrb-axis-title'}));
        children.push(text(14,top+frame.height/2,axes.labels[1],'middle',{'class':'nrb-axis-title',transform:`rotate(-90 14 ${top+frame.height/2})`}));
        host.replaceChildren(...children);
    },destroy(){host.remove();}};
}
