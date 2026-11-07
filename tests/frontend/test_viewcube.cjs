// Dependency-free ViewCube checks: node tests/frontend/test_viewcube.cjs
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const {test} = require('node:test');
const vm = require('node:vm');
const sourcePath = process.argv[2] || join(__dirname, '../../src/nurbspy/web_assets/viewcube.js');

class Element {
  constructor() {this.children=[];this.style={};this.dataset={};this.attributes={};this.events={};}
  setAttribute(key,value) {this.attributes[key]=value;}
  appendChild(child) {
    if(child.parent) child.parent.children=child.parent.children.filter(item=>item!==child);
    this.children.push(child);child.parent=this;
  }
  append(...children) {children.forEach(child=>this.appendChild(child));}
  addEventListener(name,callback) {this.events[name]=callback;}
  removeEventListener(name) {delete this.events[name];}
  focus() {this.focusCount=(this.focusCount||0)+1;}
  remove() {if(this.parent) this.parent.children=this.parent.children.filter(child=>child!==this);}
}

function browser(reducedMotion=false) {
  let now=0,next=0;
  const frames=new Map();
  const context=vm.createContext({window:{matchMedia:()=>({matches:reducedMotion})},
    document:{createElement:()=>new Element(),createElementNS:()=>new Element()},
    performance:{now:()=>now},
    requestAnimationFrame:callback=>{frames.set(++next,callback);return next;},
    cancelAnimationFrame:id=>frames.delete(id)});
  const source=readFileSync(sourcePath,'utf8').replace('export function createViewCube','function createViewCube')
    .replace(/^export \{.*\};\r?$/m,'');
  vm.runInContext(source+'\nglobalThis.cube={createViewCube,basis,targetBasis,quaternion,fromQuaternion,interpolate};',context);
  function tick(delta=50) {
    now+=delta;const pending=[...frames.values()];frames.clear();
    pending.forEach(callback=>callback(now));
  }
  function flush() {for(let count=0;frames.size&&count<30;count++) tick();assert.equal(frames.size,0);}
  return {cube:context.cube,tick,flush,frames};
}

function camera() {
  let state={position:[11,22,33],target:[10,20,30],up:[0,0,1]};
  const adapter={interactionElement:new Element(),writes:[],cancels:0,
    readCamera:()=>JSON.parse(JSON.stringify(state)),
    writeCamera(next,finished) {
      close(next.target,state.target);
      close([Math.hypot(...next.position.map((value,i)=>value-next.target[i]))],[Math.sqrt(14)]);
      state=JSON.parse(JSON.stringify(next));adapter.writes.push({state,finished});
    },
    onCancel() {adapter.cancels++;},
    setPosition(position) {state.position=position;},
  };
  return adapter;
}
function close(a,b,tolerance=1e-10) {
  assert.equal(a.length,b.length);
  a.forEach((value,i)=>assert.ok(Math.abs(value-b[i])<tolerance,`${a} differs from ${b}`));
}

test('All 26 directions produce valid camera frames for Z-up and Y-up',()=>{
  const {cube}=browser();
  for(const up of [[0,0,1],[0,1,0]]) for(const x of [-1,0,1]) for(const y of [-1,0,1]) for(const z of [-1,0,1]) {
    if(!x&&!y&&!z) continue;
    const target=cube.targetBasis([x,y,z],up);
    const restored=cube.fromQuaternion(cube.quaternion(target));
    close(restored.back,[x,y,z].map(value=>value/Math.hypot(x,y,z)));
    close(restored.up,target.up);close(restored.right,target.right);
  }
});

test('Cube exposes six faces, twelve edges, eight corners and useful face-on bevel targets',()=>{
  const {cube,flush}=browser(true),adapter=camera();
  const widget=cube.createViewCube(new Element(),adapter);
  widget.snap([-1,0,0],[0,1,0]);flush();
  const nodes=widget.host.children[0].children[1].children;
  assert.equal(nodes.length,26);
  for(const [kind,count] of [['face',6],['edge',12],['corner',8]]) {
    assert.equal(nodes.filter(node=>node.attributes['data-kind']===kind).length,count);
  }
  const visible=nodes.filter(node=>node.style.display!=='none');
  assert.equal(visible.length,9);
  for(const node of visible) {
    const points=node.children[1].attributes.points.split(' ').map(point=>point.split(',').map(Number));
    const area=Math.abs(points.reduce((sum,point,index)=>{
      const next=points[(index+1)%points.length];return sum+point[0]*next[1]-point[1]*next[0];
    },0)/2);
    assert.ok(area>50);assert.equal(node.attributes.tabindex,'0');
  }
});

test('Face, edge and corner snaps preserve focus and distance, including both poles',()=>{
  const {cube,flush}=browser(),adapter=camera();
  const widget=cube.createViewCube(new Element(),adapter);
  for(const direction of [[-1,0,0],[0,0,1],[0,0,-1],[1,1,0],[-1,-1,-1]]) {
    widget.snap(direction);flush();
    const state=adapter.readCamera();
    close(state.position.map((value,i)=>value-state.target[i]),direction.map(value=>value*Math.sqrt(14)/Math.hypot(...direction)));
    close(state.target,[10,20,30]);
    assert.equal(adapter.writes.at(-1).finished,true);
  }
  assert.ok(adapter.writes.some(row=>row.finished===false));
});

test('Camera updates are coalesced, follow external navigation and create no subscriptions',()=>{
  const {cube,flush,frames}=browser(),adapter=camera();
  const widget=cube.createViewCube(new Element(),adapter);flush();
  const before=widget.host.dataset.orientation;
  adapter.setPosition([20,20,30]);widget.update();widget.update();
  assert.equal(frames.size,1);flush();assert.notEqual(widget.host.dataset.orientation,before);
});

test('Pointer interaction cancels animation and rebases controls only once',()=>{
  const {cube,tick,flush}=browser(),adapter=camera();
  const widget=cube.createViewCube(new Element(),adapter);flush();
  widget.snap([-1,0,0]);tick();
  adapter.interactionElement.events.pointerdown();
  const stopped=adapter.readCamera();flush();
  close(adapter.readCamera().position,stopped.position);assert.equal(adapter.cancels,1);
  widget.cancel();assert.equal(adapter.cancels,1);
});

test('Destroy removes listeners and pending frames without recreating host controls',()=>{
  const {cube,flush,frames}=browser(),adapter=camera(),parent=new Element();
  const widget=cube.createViewCube(parent,adapter);widget.snap([-1,0,0]);
  widget.destroy();widget.destroy();flush();
  assert.equal(frames.size,0);assert.equal(parent.children.length,0);
  assert.equal(adapter.interactionElement.events.pointerdown,undefined);assert.equal(adapter.cancels,0);
  widget.update();widget.snap([0,0,1]);assert.equal(frames.size,0);
});

test('Reduced motion snaps in one frame and camera instances remain independent',()=>{
  const {cube,tick,flush}=browser(true),a=camera(),b=camera();
  const first=cube.createViewCube(new Element(),a);cube.createViewCube(new Element(),b);flush();
  const untouched=b.readCamera();first.snap([0,0,1]);tick();
  close(a.readCamera().position,[10,20,30+Math.sqrt(14)]);close(b.readCamera().position,untouched.position);
  assert.equal(a.writes.length,1);assert.equal(a.writes[0].finished,true);
});

test('Home restores configured engineering view while retaining focus and distance',()=>{
  const {cube,flush}=browser(true),adapter=camera();
  const widget=cube.createViewCube(new Element(),adapter);
  widget.setHome([-1,.2,.15],[0,1,0]);widget.snap([0,1,0]);flush();
  widget.host.children[1].children[0].events.click();flush();
  const expected=cube.targetBasis([-1,.2,.15],[0,1,0]),state=adapter.readCamera();
  close(state.position,expected.back.map((value,i)=>value*Math.sqrt(14)+[10,20,30][i]));
  close(state.up,expected.up);
});

test('Orbit and roll use fixed steps, and rapid animation clicks accumulate steps',()=>{
  const {cube,tick,flush}=browser(),adapter=camera();
  const widget=cube.createViewCube(new Element(),adapter);
  widget.snap([0,0,1],[0,1,0]);flush();
  widget.orbit(30,0);tick();widget.orbit(30,0);tick();widget.orbit(30,0);flush();
  close(adapter.readCamera().position,[10+Math.sqrt(14),20,30]);
  const position=adapter.readCamera().position;widget.roll(15);flush();
  close(adapter.readCamera().position,position);widget.roll(-15);flush();
  close(adapter.readCamera().up,[0,1,0]);
});

test('Keyboard navigation keeps focus off cube faces that become hidden',()=>{
  const {cube,flush}=browser(true),adapter=camera();
  const widget=cube.createViewCube(new Element(),adapter);flush();
  const svg=widget.host.children[0];
  const face=svg.children[1].children.find(node=>node.attributes['aria-label']==='View from +Z');
  face.events.keydown({key:'Enter',preventDefault(){}});flush();
  assert.equal(svg.focusCount,1);
  for(let count=0;count<3;count++) {
    svg.events.keydown({key:'ArrowRight',preventDefault(){},stopPropagation(){}});flush();
  }
  assert.equal(svg.focusCount,4);assert.equal(face.style.display,'none');
});
