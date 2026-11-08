// Linked overlays reuse GPU storage and reject results for older surface shapes.
const assert=require('node:assert/strict');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
(async()=>{
    const assets=path.resolve(__dirname,'../../src/nurbspy/web_assets');
    const {ScientificDisplay}=await import(pathToFileURL(path.join(assets,'science.js')).href);
    const THREE=await import(pathToFileURL(path.join(assets,'vendor/three.module.js')).href);
    const display=Object.create(ScientificDisplay.prototype);
    Object.assign(display,{owner:{scene:new THREE.Scene(),displayQuality:'preview'},
        isocurves:{checked:true},isocurveColor:{value:'#536d7d'},meshMode:{value:'off'}});
    const data={curves:[{direction:'u',parameter:.2,points:[[0,0,0],[1,1,1],[2,2,2]]}]};
    display.updateIsocurves(data);
    const line=display.isocurveLines[0],geometry=line.geometry,material=line.material;
    const buffer=geometry.attributes.instanceStart.data;
    data.curves[0].points[1][2]=4;
    display.updateIsocurves(data);
    assert.equal(line.geometry,geometry);assert.equal(line.material,material);
    assert.equal(geometry.attributes.instanceStart.data,buffer);
    assert.equal(geometry.attributes.instanceStart.getZ(1),4);
    assert.equal(display.isocurveGroup.children.length,1);
    console.log('PASS Isocurves update existing GPU buffers during a drag');

    let disposed=0;geometry.addEventListener('dispose',()=>disposed++);
    data.curves[0].points.push([3,3,3]);display.owner.displayQuality='full';
    display.updateIsocurves(data);
    assert.equal(disposed,1);assert.notEqual(line.geometry,geometry);
    assert.equal(display.isocurveGroup.userData.quality,'full');
    display.applyColors=()=>{};display.refresh=()=>{};
    display.geometryChanged();assert.equal(display.isocurveGroup.visible,false);
    display.clearIsocurves();assert.equal(display.owner.scene.children.length,0);
    assert.equal(display.isocurveLines.length,0);
    console.log('PASS Refinement disposes replaced buffers and hides stale overlays');

    const jobs=[],applied=[];
    const queue=Object.create(ScientificDisplay.prototype);
    Object.assign(queue,{alive:true,serial:1,running:false,pending:{revision:1,quality:'full'},
        owner:{options:{inspect:request=>new Promise(resolve=>jobs.push({request,resolve}))},render(){}},
        note:{},meshMode:{value:'off'},applyField(){},updateIsocurves:data=>applied.push(data)});
    const first=queue.flush();queue.serial=2;queue.pending={revision:2,quality:'preview'};
    jobs[0].resolve({revision:1,quality:'full',isocurves:{shape:'old'}});await first;
    assert.equal(applied.length,0);assert.equal(jobs.length,2);
    jobs[1].resolve({revision:2,quality:'preview',isocurves:{shape:'current'}});
    await new Promise(resolve=>setImmediate(resolve));
    assert.deepEqual(applied,[{shape:'current'}]);assert.equal(queue.running,false);
    console.log('PASS Older isocurve responses cannot replace the current surface overlay');
    console.log('3 isocurve display checks passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
