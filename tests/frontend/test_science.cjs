// Native values arrive from Python; verify only colouring and queue behaviour.
const assert=require('node:assert/strict');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
(async()=>{
    const url=pathToFileURL(path.resolve(__dirname,'../../src/nurbspy/web_assets/science.js')).href;
    const {fieldLimits,fieldColors,ScientificDisplay}=await import(url);
    const THREE=await import(pathToFileURL(path.resolve(__dirname,'../../src/nurbspy/web_assets/vendor/three.module.js')).href);
    let checks=0;
    function check(name,fn){fn();checks++;console.log('PASS',name);}
    check('Constant fields and manual limits retain a nonzero labelled range',()=>{
        const [low,high]=fieldLimits({range:[0,0]},null);assert.ok(low<0 && high>0);
        assert.deepEqual(fieldLimits({range:[2,4]},[-1,1]),[-1,1]);
        assert.equal(fieldLimits({range:null},null),null);
    });
    check('Colormaps reverse correctly, clamp outliers, and mark undefined samples grey',()=>{
        const table=[[0,0,0],[1,1,1]],normal=fieldColors([0,1,-5,5,null],table,[0,1]);
        assert.deepEqual(Array.from(normal.slice(0,12)),[0,0,0,1,1,1,0,0,0,1,1,1]);
        assert.deepEqual(Array.from(fieldColors([0,1],table,[0,1],true)),[1,1,1,0,0,0]);
        assert.ok(normal[12]>0 && normal[12]<1);
    });
    check('Selected-point colour applies in both projections without tinting white planar faces',()=>{
        const owner={is2D:false,pointMaterial:new THREE.MeshPhysicalMaterial(),selectedMaterial:new THREE.MeshPhysicalMaterial(),render(){}};
        const controls={owner,controlColor:{value:'#ff7f0e'},selectedColor:{value:'#8544cc'},quantity:{value:'solid'},legend:{}};
        ScientificDisplay.prototype.applyColors.call(controls);
        assert.equal(owner.selectedMaterial.color.getHexString(),'8544cc');
        assert.equal(owner.selectedMaterial.emissive.getHexString(),'8544cc');
        owner.is2D=true;owner.pointMaterial=new THREE.MeshBasicMaterial({color:'#ffffff'});
        owner.selectedMaterial=owner.pointMaterial;owner.outlineMaterial=new THREE.MeshBasicMaterial();owner.selectedOutlineMaterial=new THREE.MeshBasicMaterial();
        ScientificDisplay.prototype.applyColors.call(controls);
        assert.equal(owner.pointMaterial.color.getHexString(),'ffffff');
        assert.equal(owner.selectedOutlineMaterial.color.getHexString(),'8544cc');
    });
    check('Surface opacity survives solid/diagnostic material switches and restores depth writing',()=>{
        const solid=new THREE.MeshStandardMaterial(),field=new THREE.MeshBasicMaterial(),fill=new THREE.MeshStandardMaterial();
        const owner={geometryObject:{material:field},render(){}};
        const display={owner,isSurface:true,solidMaterial:solid,colorMaterial:field,exportFill:{material:fill},
            geometryColor:{value:'#0072bd'},controlColor:{value:'#ff7f0e'},quantity:{value:'u'},
            field:{},opacity:{value:.25},legend:{}};
        ScientificDisplay.prototype.applyColors.call(display);
        for(const material of [solid,field,fill]){
            assert.equal(material.opacity,.25);assert.equal(material.transparent,true);assert.equal(material.depthWrite,false);
        }
        display.opacity.value=1;display.quantity.value='solid';
        ScientificDisplay.prototype.applyColors.call(display);
        assert.equal(owner.geometryObject.material,solid);
        for(const material of [solid,field,fill]){assert.equal(material.opacity,1);assert.equal(material.depthWrite,true);}
    });
    check('Curve annotations can draw cleanly over surfaces without changing geometry',()=>{
        const owner={geometryObject:{material:new THREE.MeshBasicMaterial()},render(){}};
        const display={owner,isSurface:false,curveDepth:{checked:false},geometryColor:{value:'#0072bd'},
            controlColor:{value:'#ff7f0e'},quantity:{value:'solid'},legend:{}};
        ScientificDisplay.prototype.applyColors.call(display);assert.equal(owner.geometryObject.material.depthTest,false);
        assert.equal(owner.geometryObject.renderOrder,10);
        display.curveDepth.checked=true;ScientificDisplay.prototype.applyColors.call(display);
        assert.equal(owner.geometryObject.material.depthTest,true);
        assert.equal(owner.geometryObject.renderOrder,0);
    });
    const jobs=[],applied=[];
    const display=Object.create(ScientificDisplay.prototype);
    Object.assign(display,{alive:true,serial:1,running:false,pending:{revision:1,quality:'full'},
        note:{},meshMode:{value:'off'},owner:{options:{inspect:request=>new Promise(resolve=>jobs.push({request,resolve}))},render(){}},
        applyField(){applied.push(this.field?.name);}});
    const first=display.flush();assert.equal(jobs.length,1);
    display.serial=2;display.pending={revision:2,quality:'preview'};
    jobs[0].resolve({revision:1,quality:'full',field:{name:'old'}});await first;
    assert.equal(applied.length,0);assert.equal(jobs.length,2);
    jobs[1].resolve({revision:2,quality:'preview',field:{name:'new'}});
    await new Promise(resolve=>setImmediate(resolve));
    check('One inspection runs at a time and stale diagnostics cannot replace newer choices',()=>{
        assert.deepEqual(applied,['new']);assert.equal(display.running,false);
    });
    console.log(`${checks} scientific display checks passed`);
})().catch(error=>{console.error(error);process.exitCode=1;});
