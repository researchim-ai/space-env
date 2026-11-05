import * as THREE from './vendor/three.module.js';
import { OrbitControls } from './vendor/OrbitControls.js';
import { makeSatellite } from './studio.js';
import { mountWorkspace } from './workspace.js';

const $ = (id) => document.getElementById(id);
const COLORS = ['#8edfc5','#91c9ff','#f0bf80','#d6adff','#f297b1','#9bd77d'];
const actorColor=(name)=>{let h=0;for(const c of name)h=(h*31+c.charCodeAt(0))>>>0;return COLORS[h%COLORS.length];};
const reason = {observable_checks_passed:'Доступно сейчас',product_not_local:'Продукт отсутствует или уже доставлен',no_current_contact:'Нет текущего контакта',actor_busy:'Аппарат занят',role_scope:'Другая роль',control_exceeds_horizon:'Выходит за горизонт',product_not_ready:'Продукт ещё не готов',safe_mode:'Безопасный режим',accepted:'Принято',runtime_rejected:'Отклонено нативным runtime',wait:'Ожидание'};
let credentials=null, state=null, busy=false, selected=new Set(), pending=null, autorun=false;
let workspaceUI=null;
const mapImage=new Image();mapImage.src='/viz/textures/earth_day.jpg';mapImage.onload=()=>drawMap();
const fmt=(value,digits=1)=>Number(value).toLocaleString('ru-RU',{maximumFractionDigits:digits});
const say=(message,error=false)=>{$('notice').textContent=message;$('notice').classList.toggle('error',error);};
function token(admin=false){return credentials?.tokens[admin?'admin':$('role').value];}
async function api(path,{method='GET',body,admin=false,anonymous=false,binary=false,rawBody}={}){
  const headers={};if(!anonymous&&token(admin))headers.Authorization=`Bearer ${token(admin)}`;
  if(body!==undefined||rawBody!==undefined)headers['Content-Type']='application/json';
  const response=await fetch(path,{method,headers,body:rawBody??(body===undefined?undefined:JSON.stringify(body)),cache:'no-store'});
  if(!response.ok){let payload;try{payload=await response.json();}catch{payload={};}const error=new Error(payload.error?.message||payload.error||`HTTP ${response.status}`);error.status=response.status;throw error;}
  return binary?response.blob():response.json();
}
const endpoint=(suffix='')=>`/api/operator/sessions/${credentials.session_id}${suffix?'/'+suffix:''}`;
function buttons(){
  workspaceUI?.setBusy(busy);
  for(const id of ['new','import','role','preset','refresh','close'])$(id).disabled=busy;
  for(const id of ['execute','wait'])$(id).disabled=busy||!state||state.done||!state.permissions.step;
  $('run').disabled=busy&&!autorun||!state||state.done||!state.permissions.step||state.role!=='@central';$('run').textContent=autorun?'Остановить прогон':'Прогон планировщика';
  $('suggest').disabled=busy||!state||state.done||!state.permissions.step||state.role!=='@central';
  for(const id of ['close','refresh','role'])$(id).disabled=busy||!credentials;
  document.querySelectorAll('#actions input').forEach(c=>{c.disabled=busy||!state||state.done||!state.permissions.step||!state.observation.catalogue.find(a=>a.id===c.dataset.action)?.enabled;});
  for(const id of ['checkpoint','verify','export','archive','to-agent','to-operator'])$(id).disabled=busy||!credentials?.tokens.admin;
  $('connection-save').disabled=busy||!credentials;
  if(state){$('to-agent').disabled||=state.control.controller==='agent';$('to-operator').disabled||=state.control.controller==='operator';}
  $('close').disabled=busy||!credentials?.tokens.admin;
  $('workspace-result').disabled=busy||!state?.workspace;
  if(state?.workspace&&!state.workspace.plan_aligned){$('suggest').disabled=true;$('run').disabled=true;}
}
async function task(fn){if(busy)return;busy=true;buttons();try{await fn();}catch(error){say(error.message,true);if(error.status===409&&credentials)try{await refresh();}catch{} }finally{busy=false;buttons();}}
async function refresh(){if(!credentials)return;state=await api(endpoint());render();}
async function open(payload,restore=false,rawBody){
  const old=credentials;
  say(restore?'Проверяем checkpoint через native cold replay…':'Строим отдельные орбиты и открываем нативную миссию…');
  const result=await api(restore?'/api/operator/restore':'/api/operator/sessions',{method:'POST',body:payload,anonymous:true,rawBody});
  if(old&&!state?.durable)try{await fetch(`/api/operator/sessions/${old.session_id}/close`,{method:'POST',headers:{Authorization:`Bearer ${old.tokens.admin}`,'Content-Type':'application/json'},body:'{}'});}catch{}
  adopt(result);
}
function adopt(result){
  credentials=result;state=result.state;pending=null;selected.clear();globe?.reset();$('role').replaceChildren();
  for(const role of Object.keys(credentials.tokens).filter(x=>x!=='admin')){const option=document.createElement('option');option.value=role;option.textContent=role==='operator'?'Оператор · вся миссия':role==='viewer'?'Наблюдатель · только чтение':role==='agent'?'Канал агента':role.startsWith('role:')?role.slice(5):role; $('role').append(option);}
  $('role').value=credentials.tokens.operator?'operator':Object.keys(credentials.tokens).find(k=>k!=='admin');$('duration').value=state.decision_sec;$('duration').max=state.max_decision_sec;
  render();say(state.durable?'Сессия записана в durable store. Сохраните файл доступа для подключения после перезапуска.':'Сессия открыта в памяти процесса.');
}
function actionNodes(){
  const filter=$('filter').value.toLowerCase();$('actions').replaceChildren();
  for(const action of state.observation.catalogue.filter(a=>a.id!=='wait'&&(!filter||JSON.stringify([a.id,a.actors]).toLowerCase().includes(filter)))){
    const label=document.createElement('label');label.className='action'+(!action.enabled?' disabled':'');
    const checkbox=document.createElement('input');checkbox.type='checkbox';checkbox.dataset.action=action.id;checkbox.checked=selected.has(action.id);checkbox.disabled=!action.enabled||busy||!state.permissions.step||state.done;
    checkbox.addEventListener('change',()=>{if(checkbox.checked&&selected.size>=state.action_slots){checkbox.checked=false;say(`Лимит роли — ${state.action_slots} действий на шаг.`,true);return;}checkbox.checked?selected.add(action.id):selected.delete(action.id);$('selection-count').textContent=`${selected.size} / ${state.action_slots}`;});
    const text=document.createElement('span');text.textContent=action.id;const hint=document.createElement('small');hint.textContent=`${(action.actors||[]).join(', ')} · ${reason[action.reason]||action.reason}`;text.append(hint);label.append(checkbox,text);$('actions').append(label);
  }
}
function render(){
  document.body.dataset.revision=state.revision;
  $('storage').textContent=state.durable?'SQLite · подтверждённые commits':'память процесса';$('owner').textContent=`Контроль: ${state.control.controller==='agent'?'агент':'оператор'} · epoch ${state.control.epoch}`;
  $('connection').textContent='● Native session';$('revision').textContent=`rev ${state.revision}`;$('time').textContent=`${fmt(state.time_sec)} с`;$('horizon').textContent=`из ${fmt(state.horizon_sec)} с`;
  const actors=state.observation.entities.filter(e=>e.kind==='actor');$('actor-count').textContent=actors.length;$('enabled-count').textContent=state.observation.catalogue.filter(a=>a.enabled&&a.id!=='wait').length;
  $('scope-note').textContent=state.physics_profile==='reference-spacecraft-bus/1'?'Синтетическая EO/CPU/ISL/ACK-миссия со связанными ресурсами. Не калиброванный цифровой двойник. Идеальная текущая навигация; без прогноза и оценивания орбиты.':'Отдельные J2-траектории, готовые бортовые продукты и наземная доставка. Не калиброванный цифровой двойник. Идеальная текущая навигация; без прогноза.';
  $('mission-status').textContent=state.done?'Эпизод закрыт':'Ожидает шага';$('mission-name').textContent='Управление спутниковой миссией';$('session-label').textContent=`Session ${credentials.session_id} · ${state.role}`;
  const latest=state.events.at(-1);$('feedback').textContent=latest?(reason[latest.feedback.code]||latest.feedback.code):'Физика не идёт по часам браузера';
  selected=new Set([...selected].filter(id=>state.observation.catalogue.some(a=>a.id===id&&a.enabled)));$('selection-count').textContent=`${selected.size} / ${state.action_slots}`;actionNodes();
  const nav=state.frames.at(-1)?.actors||{};$('telemetry').replaceChildren();$('legend').replaceChildren();
  actors.forEach((actor,index)=>{const info=nav[actor.id],tr=document.createElement('tr');for(const text of [actor.id,`${fmt(100*actor.values.soc)}%`,`${fmt(actor.values.stored_bits/1e6,2)} Мбит`,info?`${fmt(info.alt_m/1000)} км`:'—',info?`${fmt(info.lat_deg,2)}° / ${fmt(info.lon_deg,2)}°`:'Нет навигации']){const td=document.createElement('td');td.textContent=text;tr.append(td);} $('telemetry').append(tr);const key=document.createElement('span');key.textContent=actor.id;key.style.setProperty('--color',actorColor(actor.id));$('legend').append(key);});
  $('events').replaceChildren();if(!state.events.length){const item=document.createElement('li');item.className='muted';item.textContent='Команд в журнале этой роли пока нет.';$('events').append(item);}
  for(const event of [...state.events].reverse()){const li=document.createElement('li'),title=document.createElement('b'),small=document.createElement('small');title.textContent=`${fmt(event.time_sec)} с · ${reason[event.feedback.code]||event.feedback.code}`;small.textContent=`rev ${event.revision} · ${event.action_ids.join(', ')} · Δt ${fmt(event.elapsed_sec)} с`;li.append(title,small);$('events').append(li);}
  $('products').replaceChildren();for(const entity of state.observation.entities.filter(e=>['product','request','job'].includes(e.kind))){const item=document.createElement('div');item.className='product';const title=document.createElement('b'),detail=document.createElement('small');title.textContent=entity.id;detail.textContent=`${entity.kind} · ${entity.values.remaining_bits!==undefined?fmt(entity.values.remaining_bits/1e6,3)+' Мбит осталось':(state.status_labels[String(entity.values.status_code)]||'нет статуса')}`;item.append(title,detail);$('products').append(item);}
  if(!$('products').children.length)$('products').textContent='Нет видимых продуктов или заданий.';
  const previousFocus=$('focus').value;$('focus').replaceChildren();for(const name of Object.keys(nav)){const option=document.createElement('option');option.value=name;option.textContent=name;$('focus').append(option);}if(nav[previousFocus])$('focus').value=previousFocus;
  drawMap();globe?.update(state);buttons();workspaceUI?.render(state);
}
async function step(ids){
  const advance=Number($('duration').value);if(!Number.isFinite(advance)||advance<=0)throw new Error('Задайте положительную длительность шага.');
  const request={action_ids:ids,expected_revision:state.revision,advance_sec:advance,control_epoch:state.control.epoch};
  // Preserve the key on network failure; a retry cannot execute the same step twice.
  if(!pending||JSON.stringify(pending.request)!==JSON.stringify(request))pending={request,key:crypto.randomUUID()};
  const receipt=await api(endpoint('step'),{method:'POST',body:{...request,idempotency_key:pending.key}});pending=null;
  selected.clear();await refresh();say(`${reason[receipt.feedback.code]||receipt.feedback.code} · прошло ${fmt(receipt.elapsed_sec)} с. ${receipt.feedback.accepted?'':'Отклонённое действие также расходует физическое время.'}`,!receipt.feedback.accepted);
}
function download(blob,name){const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
$('new').onclick=()=>task(()=>open({preset:$('preset').value}));$('refresh').onclick=()=>task(refresh);$('filter').oninput=()=>state&&actionNodes();$('role').onchange=()=>task(async()=>{pending=null;selected.clear();await refresh();say('Роль изменена; права и видимость проверяются сервером.');});
$('execute').onclick=()=>task(()=>step([...selected].length?[...selected]:['wait']));$('wait').onclick=()=>task(()=>step(['wait']));$('suggest').onclick=()=>task(async()=>{const suggestion=await api(endpoint('suggest'));if(suggestion.advance_sec!==undefined)$('duration').value=suggestion.advance_sec;await step(suggestion.action_ids);});
$('run').onclick=()=>{if(autorun){autorun=false;buttons();return;}task(async()=>{autorun=true;buttons();try{for(let i=0;i<256&&autorun&&state&&!state.done;i++){const suggestion=await api(endpoint('suggest'));if(suggestion.advance_sec!==undefined)$('duration').value=suggestion.advance_sec;await step(suggestion.action_ids);await new Promise(resolve=>setTimeout(resolve,120));}}finally{autorun=false;}say(state?.done?'Эпизод завершён нативной средой; итог виден в продуктах и заданиях.':'Прогон остановлен. Уже принятый шаг завершён.');});};
async function artifactDigest(blob){return [...new Uint8Array(await crypto.subtle.digest('SHA-256',await blob.arrayBuffer()))].map(x=>x.toString(16).padStart(2,'0')).join('');}
async function artifactDownload(format,filename){
 const job=await api(endpoint('artifacts'),{method:'POST',body:{direction:'download',format},admin:true});
 try{const blob=await api(`/api/operator/transfers/${job.transfer_id}/content`,{admin:true,binary:true});if(blob.size!==job.bytes||await artifactDigest(blob)!==job.sha256)throw new Error('Размер или SHA-256 скачанного артефакта не совпал.');download(blob,filename);}
 finally{try{await api(`/api/operator/transfers/${job.transfer_id}/cancel`,{method:'POST',body:{},admin:true});}catch{}}
}
async function artifactImport(file,format,takeover=false){
 const old=credentials,ephemeral=!state?.durable;
 const job=await api(endpoint('artifacts'),{method:'POST',body:{direction:'upload',format,bytes:file.size,sha256:await artifactDigest(file),takeover},admin:true});
 let result;
 try{
  for(let offset=0;offset<file.size;offset+=job.chunk_bytes){
   say(`Загрузка артефакта: ${Math.min(offset+job.chunk_bytes,file.size)} / ${file.size} байт…`);
   const response=await fetch(`/api/operator/transfers/${job.transfer_id}/chunks`,{method:'PUT',headers:{Authorization:`Bearer ${old.tokens.admin}`,'Content-Type':'application/octet-stream','X-Upload-Offset':String(offset)},body:file.slice(offset,offset+job.chunk_bytes),cache:'no-store'});
   if(!response.ok){const payload=await response.json();throw new Error(payload.error?.message||payload.error||`HTTP ${response.status}`);}
  }
  say('SHA-256 проверен. Выполняется native replay перед импортом…');
  result=await api(`/api/operator/transfers/${job.transfer_id}/commit`,{method:'POST',body:{},admin:true});
 }finally{try{await api(`/api/operator/transfers/${job.transfer_id}/cancel`,{method:'POST',body:{},admin:true});}catch{}}
 if(ephemeral)try{await fetch(`/api/operator/sessions/${old.session_id}/close`,{method:'POST',headers:{Authorization:`Bearer ${old.tokens.admin}`,'Content-Type':'application/json'},body:'{}'});}catch{}
 adopt(result);say('Артефакт импортирован после проверки SHA-256 и native replay. Выданы новые capabilities.');
}
$('checkpoint').onclick=()=>task(async()=>{await artifactDownload('checkpoint','mission-checkpoint.json');say('Checkpoint скачан без пересериализации. SHA-256 подтверждён.');});
$('verify').onclick=()=>task(async()=>{say('Пересчитываем миссию с начала; живое состояние не изменяется…');const result=await api(endpoint('verify'),{method:'POST',body:{},admin:true});$('verification').textContent=`Replay подтверждён · ${result.decisions} решений · ${result.episode_sha256.slice(0,12)}…`;say('Cold replay совпал.');});
$('export').onclick=()=>task(async()=>{say('Native replay и дисковая сборка RL/SFT…');await artifactDownload('training','mission-training.zip');say('RL/SFT экспортирован с проверкой SHA-256 и external provenance.');});
$('close').onclick=()=>task(async()=>{await api(endpoint('close'),{method:'POST',body:{},admin:true});credentials=null;state=null;pending=null;$('connection').textContent='Сессия закрыта';$('session-label').textContent='Нет активной сессии';$('mission-status').textContent='Сессия закрыта';$('actions').replaceChildren();say('Сессия закрыта. На экране последний снимок; можно открыть новую или импортировать checkpoint.');});
$('connection-save').onclick=()=>{if(credentials){const value={schema:'space-env.operator-connection/1',session_id:credentials.session_id,tokens:credentials.tokens};download(new Blob([JSON.stringify(value)],{type:'application/json'}),`operator-connection.${credentials.session_id}.json`);say('Секретный файл доступа сохранён. Не добавляйте его в репозиторий или training dataset.');}};
$('archive').onclick=()=>task(async()=>{await artifactDownload('archive','mission-archive.json');say('Архив скачан. SHA-256 подтверждён; capabilities в архив не включены.');});
async function transfer(controller){const request={controller,expected_revision:state.revision,expected_control_epoch:state.control.epoch,idempotency_key:crypto.randomUUID(),reason:$('handover-reason').value};await api(endpoint('handover'),{method:'POST',body:request,admin:true});pending=null;await refresh();say('Управление передано. Физическое время не изменилось; команды прежнего control epoch отклоняются.');}
$('to-agent').onclick=()=>task(()=>transfer('agent'));$('to-operator').onclick=()=>task(()=>transfer('operator'));
$('import').onclick=()=>$('upload').click();$('upload').onchange=()=>task(async()=>{const file=$('upload').files[0];if(!file)return;if(file.size>32*1024*1024)throw new Error('Лимит артефакта — 32 МиБ.');const raw=await file.text();const data=JSON.parse(raw);
if(data.schema==='space-env.operator-connection/1'){
 if(!/^[A-Za-z0-9_-]{20,64}$/.test(data.session_id)||!data.tokens||!Object.keys(data.tokens).some(k=>k!=='admin'))throw new Error('Некорректный файл доступа.');
 const previous=credentials;credentials={session_id:data.session_id,tokens:data.tokens};$('role').replaceChildren();const key=data.tokens.operator?'operator':Object.keys(data.tokens).find(k=>k!=='admin');const option=document.createElement('option');option.value=key;$('role').append(option);$('role').value=key;
 try{const restored=await api(endpoint());adopt({...credentials,state:restored});say('Подключение восстановлено к подтверждённой сессии.');}catch(error){credentials=previous;if(previous&&state)adopt({...previous,state});else{$('role').replaceChildren();}throw error;}
}else if(data.schema==='space-env.operator-archive/1'){if(credentials?.tokens.admin)await artifactImport(file,'archive');else if(file.size<=8*1024*1024)await open({archive:data},true,`{"archive":${raw}}`);else throw new Error('Для большого импорта подключите admin-доступ к сессии или откройте новую миссию.');}
else if(data.schema?.startsWith('space-env.task-episode/')){if(credentials?.tokens.admin)await artifactImport(file,'checkpoint',data.task?.policy_origin!=='external');else if(file.size<=8*1024*1024)await open({record:data},true,`{"record":${raw},"takeover":${data.task?.policy_origin!=='external'}}`);else throw new Error('Для большого импорта нужен admin-доступ к сессии.');}
else await open(data);
$('upload').value='';});


function drawMap(){if(!state)return;const canvas=$('ground-map'),width=canvas.clientWidth,height=canvas.clientHeight,dpr=window.devicePixelRatio||1;canvas.width=width*dpr;canvas.height=height*dpr;const c=canvas.getContext('2d');c.scale(dpr,dpr);c.fillStyle='#091320';c.fillRect(0,0,width,height);if(mapImage.complete&&mapImage.naturalWidth){c.globalAlpha=.45;c.drawImage(mapImage,0,0,width,height);c.globalAlpha=1;}c.strokeStyle='#91afcc30';c.lineWidth=1;for(let lon=-180;lon<=180;lon+=30){const x=(lon+180)/360*width;c.beginPath();c.moveTo(x,0);c.lineTo(x,height);c.stroke();}for(let lat=-90;lat<=90;lat+=30){const y=(90-lat)/180*height;c.beginPath();c.moveTo(0,y);c.lineTo(width,y);c.stroke();}const point=(p)=>[(p.lon_deg+180)/360*width,(90-p.lat_deg)/180*height];const boxes=[];function label(text,x,y,color){c.font='11px system-ui';const w=c.measureText(text).width;for(const [dx,dy] of [[8,-11],[8,20],[8,-26],[8,35],[-w-8,-11],[-w-8,20]]){const left=Math.max(2,Math.min(width-w-2,x+dx)),top=Math.max(12,Math.min(height-3,y+dy));if(boxes.some(b=>left<b.x+b.w&&left+w>b.x&&top-11<b.y+13&&top>b.y))continue;boxes.push({x:left,y:top-11,w:w+4});c.lineWidth=3;c.strokeStyle='#06101b';c.strokeText(text,left,top);c.fillStyle=color;c.fillText(text,left,top);return;}}
  for(const s of state.stations){const [x,y]=point(s);c.fillStyle='#ffcc8b';c.fillRect(x-3,y-3,6,6);label(s.name,x,y,'#ffcc8b');}
  Object.keys(state.frames.at(-1)?.actors||{}).forEach((name,index)=>{const color=actorColor(name);c.strokeStyle=color;c.lineWidth=1.7;let previous=null;for(const frame of state.frames){const p=frame.actors[name];if(!p)continue;const [x,y]=point(p);if(previous){let [px,py]=previous;const dx=x-px;if(Math.abs(dx)<width/2){c.beginPath();c.moveTo(px,py);c.lineTo(x,y);c.stroke();}else{const adjusted=x+(dx>0?-width:width),edge=dx>0?0:width,ratio=(edge-px)/(adjusted-px),ey=py+ratio*(y-py);c.beginPath();c.moveTo(px,py);c.lineTo(edge,ey);c.moveTo(width-edge,ey);c.lineTo(x,y);c.stroke();}}previous=[x,y];}if(previous){const [x,y]=previous;c.fillStyle=color;c.beginPath();c.arc(x,y,4,0,Math.PI*2);c.fill();label(name,x,y,color);}});
}
function mountGlobe(){const host=$('globe'),renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));host.append(renderer.domElement);const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(40,1,.05,100);camera.position.set(3.2,1.6,3.4);const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.minDistance=1.5;controls.maxDistance=8;scene.add(new THREE.AmbientLight('#c9e7ff',2));const light=new THREE.DirectionalLight('#ffffff',3);light.position.set(3,5,4);scene.add(light);const loader=new THREE.TextureLoader(),texture=loader.load('/viz/textures/earth_day.jpg');texture.colorSpace=THREE.SRGBColorSpace;scene.add(new THREE.Mesh(new THREE.SphereGeometry(1,72,48),new THREE.MeshBasicMaterial({map:texture})));scene.add(new THREE.Mesh(new THREE.SphereGeometry(1.008,24,16),new THREE.MeshBasicMaterial({color:'#7cbfff',wireframe:true,transparent:true,opacity:.05})));const models=new Map(),lines=new THREE.Group(),stations=new THREE.Group();scene.add(lines,stations);let initial=true;const pos=(r)=>new THREE.Vector3(r[0],r[2],-r[1]).multiplyScalar(1/6378137);function clear(group){for(const item of [...group.children]){group.remove(item);item.geometry?.dispose();item.material?.dispose();}}new ResizeObserver(()=>{const w=host.clientWidth,h=host.clientHeight;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();}).observe(host);renderer.setAnimationLoop(()=>{controls.update();renderer.render(scene,camera);});return{reset(){initial=true;},focus(name){const model=models.get(name);if(model?.visible){camera.position.copy(model.position.clone().normalize().multiplyScalar(3.6)).add(new THREE.Vector3(0,1.2,0));controls.target.set(0,0,0);controls.update();}},update(data){clear(lines);clear(stations);for(const model of models.values())model.visible=false;Object.entries(data.frames.at(-1)?.actors||{}).forEach(([name,nav],i)=>{let model=models.get(name);if(!model){model=makeSatellite(actorColor(name));model.scale.setScalar(1.6);scene.add(model);models.set(name,model);}model.visible=true;model.position.copy(pos(nav.r_itrs_m));model.lookAt(0,0,0);const points=data.frames.filter(f=>f.actors[name]).map(f=>pos(f.actors[name].r_itrs_m));if(points.length>1)lines.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color:actorColor(name),transparent:true,opacity:.75})));if(initial){camera.position.copy(pos(nav.r_itrs_m).normalize().multiplyScalar(3.6)).add(new THREE.Vector3(0,1.2,0));initial=false;}});for(const station of data.stations){const marker=new THREE.Mesh(new THREE.SphereGeometry(.012,12,8),new THREE.MeshBasicMaterial({color:'#ffcc8b'}));marker.position.copy(pos(station.r_itrs_m));stations.add(marker);}}};}
let globe=null;try{globe=mountGlobe();}catch{ $('globe-note').textContent='WebGL недоступен: управление, телеметрия и 2D-карта работают независимо.'; }
$('focus').onchange=()=>globe?.focus($('focus').value);
new ResizeObserver(drawMap).observe($('ground-map'));
// Browser/tab closure never closes a durable mission or stores secrets in localStorage.
workspaceUI=mountWorkspace({api,task,adopt,say,download});
$('workspace-result').onclick=()=>{if(state?.workspace)download(new Blob([JSON.stringify({schema:'space-env.mission-result/1',time_sec:state.time_sec,...state.workspace},null,2)],{type:'application/json'}),'mission-result.json');};
task(async()=>{await workspaceUI.initialize();const catalogue=await api('/api/operator/catalogue',{anonymous:true});$('connection').textContent='Готово к запуску';say(catalogue.active_sessions?'Есть сохранённые сессии: импортируйте файл доступа или создайте новую миссию.':'Настройте миссию выше или оставьте пример и нажмите «Составить план и открыть миссию».');});
