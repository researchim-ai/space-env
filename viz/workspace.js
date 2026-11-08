// Mission editing and planned/actual results share the native operator page.
const $=id=>document.getElementById(id);
const labels={released:'Ожидает выполнения',pending:'Ожидает выпуска',running:'В работе',completed:'Доставлена',expired:'Срок истёк',failed:'Не выполнена',cancelled:'Отменена',unreleased:'Ещё не выпущена'};
const stages={slew:'Наведение',imaging:'Съёмка',transform:'Обработка',downlink:'Передача'};
function element(tag,text,className){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(className)e.className=className;return e;}
export function mountWorkspace({api,task,adopt,say,download}){
 let configuration=null,busy=false,sessionId=null;
 function field(label,value,change,{type='number',step='any',min,max,options}={}){
  const wrap=element('label',label);let input;
  if(options){input=document.createElement('select');for(const [id,title] of options){const option=element('option',title);option.value=id;input.append(option);}if(!options.some(x=>x[0]===String(value??''))){const option=element('option',`${value} · не найден`);option.value=value;input.append(option);}}
  else{input=document.createElement('input');input.type=type;if(type==='number'){input.step=step;if(min!==undefined)input.min=min;if(max!==undefined)input.max=max;}}
  input.value=value??'';input.required=!options?.some(x=>x[0]==='');input.onchange=()=>change(type==='number'&&!options?Number(input.value):input.value);wrap.append(input);return wrap;
 }
 function actors(){return configuration.planes.flatMap(p=>p.phases_deg.map((_,i)=>`${p.id}-${String(i).padStart(2,'0')}`));}
 function removeButton(fn){const b=element('button','Удалить','secondary small');b.type='button';b.onclick=fn;return b;}
 function tables(){
  if(!configuration)return;
  $('mission-planes').replaceChildren();configuration.planes.forEach((p,i)=>{
   const row=element('div',undefined,'edit-row orbit-row');
   row.append(field('Орбита / ID',p.id,v=>{p.id=v;tables();},{type:'text'}),field('Высота от R⊕, км',(p.radius_m-6378137)/1000,v=>p.radius_m=6378137+1000*v,{min:322,max:1121}),field('Наклонение, °',p.inclination_deg,v=>p.inclination_deg=v,{min:0,max:180}),field('RAAN, °',p.raan_deg,v=>p.raan_deg=v,{min:-360,max:360}),field('Фазы, °',p.phases_deg.join(', '),v=>{p.phases_deg=v.split(',').map(s=>Number(s.trim()));tables();},{type:'text'}),removeButton(()=>{configuration.planes.splice(i,1);tables();}));$('mission-planes').append(row);
  });
  $('mission-stations').replaceChildren();configuration.stations.forEach((s,i)=>{
   const row=element('div',undefined,'edit-row station-row');
   row.append(field('Станция / ID',s.name,v=>{s.name=v;tables();},{type:'text'}),field('Широта, °',s.lat_deg,v=>s.lat_deg=v,{min:-90,max:90}),field('Долгота, °',s.lon_deg,v=>s.lon_deg=v,{min:-180,max:180}),field('Терминалы',s.terminals,v=>s.terminals=v,{min:1,max:4,step:1}),removeButton(()=>{configuration.stations.splice(i,1);tables();}));$('mission-stations').append(row);
  });
  $('mission-orders').replaceChildren();configuration.requests.forEach((r,i)=>{
   const row=element('div',undefined,'edit-row order-row');
   row.append(field('Заявка / ID',r.id,v=>r.id=v,{type:'text'}),field('Спутник',r.actor_id,v=>r.actor_id=v||null,{options:[['','Автоматически'],...actors().map(a=>[a,a])]}),field('Широта цели, °',r.lat_deg,v=>r.lat_deg=v,{min:-90,max:90}),field('Долгота цели, °',r.lon_deg,v=>r.lon_deg=v,{min:-180,max:180}),field('Станция',r.station,v=>r.station=v||null,{options:[['','Автоматически'],...configuration.stations.map(s=>[s.name,s.name])]}),field('Выпуск, с',r.release_sec,v=>r.release_sec=v,{min:0}),field('Дедлайн, с',r.deadline_sec,v=>r.deadline_sec=v,{min:1,max:3600}),field('Приоритет',r.priority,v=>r.priority=v,{min:0,max:10000,step:1}),removeButton(()=>{configuration.requests.splice(i,1);tables();}));$('mission-orders').append(row);
  });
  $('mission-size').textContent=`${actors().length} спутн. · ${configuration.stations.length} станц. · ${configuration.requests.length} заявок`;
  for(const [id,limit,count] of [['add-plane',4,actors().length],['add-station',16,configuration.stations.length],['add-order',64,configuration.requests.length]])$(id).disabled=busy||count>=limit;
  $('configuration-json').value=JSON.stringify(configuration,null,2);
 }
 function fill(value){configuration=structuredClone(value);for(const [id,key] of [['mission-title-input','name'],['mission-horizon','horizon_sec'],['mission-step','decision_sec'],['mission-memory','memory_mbit'],['mission-battery','battery_wh'],['mission-rate','rate_bps'],['mission-epoch','epoch_utc']])$(id).value=configuration[key];tables();}
 function read(){
  if(!$('mission-form').reportValidity())throw new Error('Заполните обязательные поля и проверьте диапазоны.');
  for(const [id,key] of [['mission-title-input','name'],['mission-horizon','horizon_sec'],['mission-step','decision_sec'],['mission-memory','memory_mbit'],['mission-battery','battery_wh'],['mission-rate','rate_bps'],['mission-epoch','epoch_utc']])configuration[key]=['name','epoch_utc'].includes(key)?$(id).value:Number($(id).value);
  return structuredClone(configuration);
 }
 async function create(value){say('Рассчитываем окна наблюдения и связи, распределяем задачи…');const result=await api('/api/operator/workspace',{method:'POST',body:value,anonymous:true});adopt(result);$('mission-board').scrollIntoView({behavior:'smooth',block:'start'});say(`Миссия открыта. План: ${result.state.workspace.planned} из ${result.state.workspace.total}. Нажмите «Прогон планировщика» или выполняйте шаги вручную.`);}
 $('network-open').onclick=()=>task(async()=>{const value=await api('/api/operator/network/preset',{anonymous:true});$('network-json').value=JSON.stringify(value,null,2);$('network-config').open=true;$('network-config').scrollIntoView({behavior:'smooth',block:'center'});say('Сетевой профиль: проверьте JSON и постройте совместный план. Он создаст отдельную миссию.');});
 $('network-create').onclick=()=>task(async()=>{const value=JSON.parse($('network-json').value);say('Рассчитываем EO-окна, sampled RF и маршруты custody…');const result=await api('/api/operator/network',{method:'POST',body:value,anonymous:true});adopt(result);$('mission-board').scrollIntoView({behavior:'smooth',block:'start'});say(`Сетевая миссия открыта. Прогноз: ${result.state.network.planned}/${result.state.network.total}. Запустите план для подтверждения результата.`);});
 $('mission-form').onsubmit=e=>{e.preventDefault();try{const value=read();task(()=>create(value));}catch(error){say(error.message,true);}};
 $('mission-horizon').oninput=()=>{if(!configuration)return;configuration.horizon_sec=Number($('mission-horizon').value);for(const r of configuration.requests)r.deadline_sec=Math.min(r.deadline_sec,configuration.horizon_sec);if(configuration.horizon_sec/Number($('mission-step').value)>240){$('mission-step').value=[5,10,15,20,30].find(q=>configuration.horizon_sec/q<=240&&configuration.horizon_sec%q===0)||30;}configuration.decision_sec=Number($('mission-step').value);tables();};
 $('add-plane').onclick=()=>{const index=configuration.planes.length;let id=`sat${index+1}`;while(configuration.planes.some(p=>p.id===id))id+='x';configuration.planes.push({id,radius_m:7000000,inclination_deg:55+index*5,raan_deg:60*index,phases_deg:[0]});tables();};
 $('add-station').onclick=()=>{let name=`Ground${configuration.stations.length+1}`;while(configuration.stations.some(s=>s.name===name))name+='x';configuration.stations.push({name,lat_deg:0,lon_deg:30,alt_m:0,min_elevation_deg:5,terminals:1});tables();};
 $('add-order').onclick=()=>{let id=`order${configuration.requests.length+1}`;while(configuration.requests.some(r=>r.id===id))id+='x';configuration.requests.push({id,actor_id:null,station:null,lat_deg:0,lon_deg:0,priority:1,release_sec:0,deadline_sec:Number($('mission-horizon').value),acquisition_sec:2,processing_sec:1,raw_bits:1000000,output_bits:500000});tables();};
 $('configuration-save').onclick=()=>task(async()=>{const value=read();download(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}),'mission-configuration.json');say('Конфигурация сохранена без токенов доступа.');});
 $('configuration-apply').onclick=()=>task(async()=>{const input=JSON.parse($('configuration-json').value);const {configuration:value}=await api('/api/operator/workspace/configuration',{method:'POST',body:input,anonymous:true});fill(value);say('JSON применён к редактору. Миссия изменится только после создания нового плана.');});
 $('mission-default').onclick=()=>task(async()=>{const value=await api('/api/operator/workspace/presets',{anonymous:true});fill(value.configuration);say('Загружен пример двух спутников и двух заявок.');});
 $('mission-auto').onclick=()=>task(async()=>{const value=await api('/api/operator/workspace/presets',{anonymous:true});for(const r of value.configuration.requests){r.actor_id=null;r.station=null;}fill(value.configuration);say('Автоназначение: планировщик сравнит пары спутник/станция. Ручной выбор можно закрепить отдельно для каждой заявки.');});
 $('mission-outage').onclick=()=>task(async()=>{const value=await api('/api/operator/workspace/presets',{anonymous:true});value.configuration.outages=[{station:'West',start_sec:0,end_sec:value.configuration.horizon_sec}];fill(value.configuration);say('Пример: West недоступна весь горизонт. План покажет заявку без окна связи.');});
 return {
  async initialize(){const value=await api('/api/operator/workspace/presets',{anonymous:true});fill(value.configuration);const contracts=await api('/api/operator/workspace/physics',{anonymous:true});for(const profile of contracts.profiles){const card=element('article',undefined,'physics-card');card.append(element('b',profile.id),element('p',profile.models.join(' · ')),element('p','Границы: '+profile.limits.join(' · '),'muted small'));$('physics-contracts').append(card);}},
  setBusy(value){busy=value;for(const id of ['network-open','network-create','network-json'])$(id).disabled=value;$('mission-form').querySelectorAll('input,select,button,textarea').forEach(e=>e.disabled=value);for(const id of ['mission-default','mission-auto','mission-outage'])$(id).disabled=value;if(configuration)for(const [id,limit,count] of [['add-plane',4,actors().length],['add-station',16,configuration.stations.length],['add-order',64,configuration.requests.length]])$(id).disabled=value||count>=limit;},
  render(state){
   const network=state?.network;$('network-results').hidden=!network;
   if(network){$('network-change-note').textContent=network.dynamic?`Ветка из ${network.dynamic.prefix_time_sec} с; после native-событий ${network.dynamic.committed_time_sec} с. Источник сохранён.`:'Изменения доступны центральному оператору до конца миссии.';$('network-title').textContent=network.name;$('network-count').textContent=`Доставлено ${network.fulfilled} / ${network.total}`;$('network-status').textContent=`Прогноз: ${network.planned}/${network.total} · поиск ${network.search_nodes} узлов${network.budget_exhausted?' · бюджет исчерпан':''}${!network.plan_aligned?' · план отклонён, выполните пересчёт оставшихся этапов':''}. Только native ACK подтверждает доставку.`;$('network-orders').replaceChildren();for(const r of network.requests){const card=element('article',undefined,'request-card');card.dataset.status=r.status;card.append(element('b',r.id),element('span',labels[r.status]||r.status,'badge'));if(r.plan){card.append(element('p',`${r.plan.path.join(' → ')} → ${r.plan.station}`));card.append(element('p',r.plan.stages.map(s=>`${s.kind==='relay'?'ISL':stages[s.kind]} @ ${s.cell*state.decision_sec} с`).join(' · '),'muted small'));}if(r.plan_reason||r.reason)card.append(element('p',r.reason||r.plan_reason,'request-warning'));$('network-orders').append(card);}if(!network.plan_aligned){$('suggest').disabled=true;$('run').disabled=true;}document.querySelectorAll('[data-workflow-step]').forEach(e=>e.dataset.active=Number(e.dataset.workflowStep)===(state.done?4:state.revision?3:2)?'true':'false');}
   const w=state?.workspace;$('mission-results').hidden=!w;
   if(!w)return;
   if(sessionId!==state.session_id){sessionId=state.session_id;fill(w.configuration);}
   $('workspace-name').textContent=w.name;$('workspace-count').textContent=`Доставлено ${w.fulfilled} / ${w.total}`;
   $('workspace-plan').textContent=`Прогноз плана: ${w.planned} / ${w.total} · исходная эвристика: ${w.planner.baseline.predicted_fulfilled} · поиск: ${((w.replan||w.dynamic)?w.planner.search_nodes:(w.assignment?.search_nodes??w.planner.search_nodes)).toLocaleString('ru-RU')} узлов`;
   $('workspace-outcome').textContent=!w.plan_aligned?'План отклонён от исполненной истории. Доступно ручное управление; для нового расписания используйте перепланирование на допустимой границе.':w.done?(w.all_fulfilled?'Все заявки доставлены.':`Миссия завершена. Не выполнено ${w.total-w.fulfilled} из ${w.total}; причины указаны ниже.`):'План — прогноз. Статусы и отметки этапов ниже отражают только исполненную физику.';
   if(!w.plan_aligned){$('suggest').disabled=true;$('run').disabled=true;}
   if(w.assignment?.automatic&&!w.replan&&!w.dynamic)$('workspace-plan').textContent+=` · автоназначение: ${w.assignment.trials.length} вариантов`;
   $('workspace-replan-note').textContent=(w.dynamic?`События применены на ${w.dynamic.committed_time_sec} с; исходная ветка сохранена. `:'')+w.replan_note+(w.replan?` · Ветка от ${w.replan.prefix_time_sec} с, сохранено завершённых заявок: ${w.replan.completed.length}.`:'');
   $('workspace-orders').replaceChildren();
   for(const request of w.requests){
    const card=element('article',undefined,'request-card');card.dataset.status=request.status;
    const head=element('div',undefined,'request-head');head.append(element('b',request.id),element('span',labels[request.status]||request.status,'badge'));card.append(head);
    card.append(element('p',`${request.actor} → ${request.station} · приоритет ${request.priority} · дедлайн ${request.deadline_sec} с`,'muted small'));
    const timeline=element('div',undefined,'request-stages');
    for(const stage of request.stages){const node=element('span',`${stages[stage.kind]} · план ${stage.at_sec} с`);if(request.actual_stages?.[stage.kind]){node.className='stage-done';node.textContent+=' · ✓ факт';}timeline.append(node);}card.append(timeline);
    if(request.plan_reason)card.append(element('p',request.plan_reason,'request-warning'));
    if(request.reason)card.append(element('p',`Фактическая причина: ${request.reason}`,'request-warning'));
    $('workspace-orders').append(card);
   }
   document.querySelectorAll('[data-workflow-step]').forEach(e=>e.dataset.active=Number(e.dataset.workflowStep)===(w.done?4:state.revision?3:2)?'true':'false');
  }
 };
}
