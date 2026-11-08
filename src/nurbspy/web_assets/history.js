/* Session-local geometry history. Never stores sampled meshes or camera moves. */
const clone=value=>JSON.parse(JSON.stringify(value));

export class EditHistory {
    constructor({restore,busy=()=>false,changed=()=>{},limit=100,bytes=16*1024*1024}={}){
        Object.assign(this,{restore,busy,changed,limit,bytes});
        this.past=[];this.future=[];this.transaction=null;this.restoring=false;this.serial=0;
    }
    begin(object,model,selection){
        if(this.transaction||this.restoring||this.busy())return false;
        this.transaction={token:++this.serial,object,before:clone(model),selection:clone(selection),committing:false};
        this.changed();return true;
    }
    finish(model){
        const transaction=this.transaction;if(!transaction)return;
        this.transaction=null;
        if(JSON.stringify(transaction.before)!==JSON.stringify(model)){
            const entry={object:transaction.object,before:transaction.before,after:clone(model),selection:transaction.selection};
            entry.bytes=JSON.stringify(entry).length*2;
            this.past.push(entry);this.future=[];
            let size=this.past.reduce((total,item)=>total+item.bytes,0);
            while(this.past.length>this.limit||size>this.bytes)size-=this.past.shift().bytes;
        }
        this.changed();
    }
    cancel(){this.transaction=null;this.changed();}
    get locked(){return Boolean(this.transaction||this.restoring||this.busy());}
    get canUndo(){return !this.locked&&this.past.length>0;}
    get canRedo(){return !this.locked&&this.future.length>0;}
    async travel(redo=false){
        if(redo?!this.canRedo:!this.canUndo)return false;
        const source=redo?this.future:this.past,target=redo?this.past:this.future,entry=source.at(-1);
        this.restoring=true;this.changed();
        try{
            await this.restore(entry.object,clone(redo?entry.after:entry.before),clone(entry.selection));
            source.pop();target.push(entry);return true;
        }finally{this.restoring=false;this.changed();}
    }
}
