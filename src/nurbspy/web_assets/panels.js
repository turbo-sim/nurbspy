/* One inspector shell for display and optional editing controls in every host. */
function element(tag,attributes={},text){
    const node=document.createElement(tag);
    for(const [name,value] of Object.entries(attributes))node.setAttribute(name,value);
    if(text!==undefined)node.textContent=text;
    return node;
}

export function createInspector(owner,kind,title,label){
    const sidebar=element('aside',{class:'nrb-sidebar',id:`${owner.root.id}-${kind}-settings`,'aria-label':`${title} settings`,hidden:''});
    const heading=element('div',{class:'nrb-panel-heading'});
    heading.append(element('h4',{},title));
    const close=element('button',{type:'button','aria-label':`Close ${label}`,title:`Close ${label}`},'×');heading.append(close);
    const body=element('div',{class:'nrb-config-panel'});sidebar.append(heading,body);
    const toggle=element('button',{type:'button',class:'nrb-panel-toggle','aria-expanded':'false','aria-controls':sidebar.id,'aria-label':`Open ${label}`,title:`Open ${label}`},title);
    toggle.addEventListener('click',()=>owner.setPanelOpen(owner.activePanel!==kind,kind));
    close.addEventListener('click',()=>{owner.setPanelOpen(false,kind);toggle.focus();});
    return {kind,label,sidebar,body,toggle};
}

export function syncInspector(panel,active){
    const open=panel.kind===active;
    panel.sidebar.hidden=!open;panel.toggle.setAttribute('aria-expanded',String(open));
    panel.toggle.setAttribute('aria-label',`${open?'Close':'Open'} ${panel.label}`);
    panel.toggle.title=`${open?'Close':'Open'} ${panel.label}`;
}
