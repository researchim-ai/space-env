// All visual motion is illustrative. The native simulator runs separately in Python.
const menu = document.querySelector('.menu-toggle');
menu?.addEventListener('click', () => { const open = menu.getAttribute('aria-expanded') !== 'true'; menu.setAttribute('aria-expanded', String(open)); menu.setAttribute('aria-label', open ? 'Закрыть меню' : 'Открыть меню'); document.querySelector('#navigation').classList.toggle('open', open); });
document.addEventListener('keydown', event => { if(event.key==='Escape' && menu?.getAttribute('aria-expanded')==='true'){menu.click();menu.focus();} });
const toast = document.querySelector('.toast'); let toastTimer;
function notify(message) { toast.textContent = message; toast.classList.add('visible'); clearTimeout(toastTimer); toastTimer = setTimeout(() => toast.classList.remove('visible'), 2500); }
for (const button of document.querySelectorAll('.copy')) button.addEventListener('click', async () => {
  const text = button.closest('.code-block').querySelector('code').textContent;
  try { if (!navigator.clipboard?.writeText) throw new Error('Clipboard unavailable'); await navigator.clipboard.writeText(text); notify('Команда скопирована'); }
  catch { const selection = window.getSelection(); const range = document.createRange(); range.selectNodeContents(button.closest('.code-block').querySelector('code')); selection.removeAllRanges(); selection.addRange(range); notify('Текст выделен. Нажмите Ctrl+C / ⌘C.'); }
});
const scenarios = {
 relay: {tag:'EO → CPU → ISL → GROUND ACK', title:'Снимок — это только начало.', description:'Аппарат получает данные, обрабатывает их на борту и передаёт через ретранслятор. Результат засчитывается после наземного подтверждения, а не после отправки первого фрагмента.', decisions:'Съёмка, CPU, маршрут, передача', limits:'Энергия, память, TTL, контакт', stages:[['01 / SAT-A','Наблюдение','Raw product'],['02 / ONBOARD','Обработка','CPU + energy + scratch memory'],['03 / SAT-B','Ретрансляция','Custody + fragments + TTL'],['04 / GROUND','Подтверждение','Fulfilled request']]},
 resource: {tag:'FLEET → SHARED TERMINALS → DELIVERY', title:'Окно одно. Задач — несколько.', description:'Аппараты конкурируют за наземные терминалы и ограниченные бортовые ресурсы. Контроллер выбирает совместимый набор операций: разрешённые по отдельности действия ещё не гарантируют допустимый совместный шаг.', decisions:'Приоритет, совместный batch, ожидание', limits:'Терминалы, reservations, SOC, дедлайны', stages:[['01 / FLEET','Заявки группировки','Products + priorities'],['02 / CONTROL','Выбор действий','Roles + static claims'],['03 / GROUND','Общие терминалы','Reservations + native admission'],['04 / MISSION','Доставка данных','Acknowledged ground products']]},
 recovery: {tag:'OUTAGE / ACK LOSS / TTL → REPLAN', title:'Сбой меняет маршрут. Не правила.', description:'Потеря подтверждения, недоступная станция или истекающий TTL требуют нового решения. Отмена не возвращает потраченную энергию, а неудачная операция остаётся частью физической истории.', decisions:'Повторная попытка, отмена, reroute', limits:'Оставшийся TTL, ресурсы, causal knowledge', stages:[['01 / NETWORK','Нарушение доставки','Outage / ACK loss'],['02 / EVIDENCE','Доступный feedback','No future-fault oracle'],['03 / POLICY','Перепланирование','Declared alternate routes'],['04 / OUTCOME','Результат миссии','Delivered / expired / incomplete']]}
};
function selectScenario(button) {
 const key=button.dataset.scenario, data=scenarios[key];
 for(const tab of document.querySelectorAll('[data-scenario]')){ const active=tab===button;tab.setAttribute('aria-selected',String(active));tab.tabIndex=active?0:-1; }
 document.querySelector('#scenario-panel').setAttribute('aria-labelledby',button.id);
 for(const [id,value] of Object.entries({'scenario-tag':data.tag,'scenario-title':data.title,'scenario-description':data.description})) document.getElementById(id).textContent=value;
 const facts=document.querySelectorAll('#scenario-facts b');facts[0].textContent=data.decisions;facts[1].textContent=data.limits;
 const pipeline=document.querySelector('#pipeline');pipeline.replaceChildren();
 data.stages.forEach((stage,index)=>{if(index){const arrow=document.createElement('i');arrow.textContent='↓';arrow.setAttribute('aria-hidden','true');pipeline.append(arrow);}const node=document.createElement('div');for(const [i,tag] of ['span','b','small'].entries()){const element=document.createElement(tag);element.textContent=stage[i];node.append(element);}pipeline.append(node);});
}
const tabs=[...document.querySelectorAll('[data-scenario]')];
for(const tab of tabs){tab.addEventListener('click',()=>selectScenario(tab));tab.addEventListener('keydown',event=>{let index=tabs.indexOf(tab);if(event.key==='ArrowRight')index=(index+1)%tabs.length;else if(event.key==='ArrowLeft')index=(index+tabs.length-1)%tabs.length;else if(event.key==='Home')index=0;else if(event.key==='End')index=tabs.length-1;else return;event.preventDefault();selectScenario(tabs[index]);tabs[index].focus();});}
const observedSections=[...document.querySelectorAll('.doc-content section[id]')];
if(observedSections.length && 'IntersectionObserver' in window){const observer=new IntersectionObserver(entries=>{const visible=entries.filter(e=>e.isIntersecting).sort((a,b)=>a.boundingClientRect.top-b.boundingClientRect.top);if(visible.length)for(const link of document.querySelectorAll('.toc a')){const active=link.hash==='#'+visible[0].target.id;link.classList.toggle('current',active);if(active)link.setAttribute('aria-current','location');else link.removeAttribute('aria-current');}},{rootMargin:'-5% 0px -65% 0px'});observedSections.forEach(section=>observer.observe(section));}
const canvas=document.getElementById('orbit');
if(canvas) mountOrbit(canvas);
function mountOrbit(canvas){
 const ctx=canvas.getContext('2d');if(!ctx)return;
 const reduced=matchMedia('(prefers-reduced-motion: reduce)');let running=!reduced.matches,frame=0,last=0,phase=.8,width=500,height=400,earth=null,visible=true;
 const toggle=document.getElementById('orbit-toggle');
 const updateToggle=()=>{toggle.textContent=running?'Ⅱ':'▷';toggle.setAttribute('aria-label',running?'Приостановить анимацию':'Продолжить анимацию');toggle.setAttribute('aria-pressed',String(!running));};
 toggle.addEventListener('click',()=>{running=!running;updateToggle();schedule();});updateToggle();
 const image=new Image();image.src=new URL('./earth.jpg',import.meta.url).href;
 image.onload=()=>{const source=document.createElement('canvas');source.width=image.width;source.height=image.height;const s=source.getContext('2d',{willReadFrequently:true});s.drawImage(image,0,0);const pixels=s.getImageData(0,0,source.width,source.height).data;
  const size=500,sphere=document.createElement('canvas');sphere.width=size;sphere.height=size;const c=sphere.getContext('2d'),out=c.createImageData(size,size);
  for(let y=0;y<size;y++)for(let x=0;x<size;x++){const nx=(x-size/2)/(size/2),ny=(y-size/2)/(size/2),r2=nx*nx+ny*ny;if(r2>1)continue;const z=Math.sqrt(1-r2),lat=Math.asin(-ny),lon=Math.atan2(nx,z)-.55;const u=((lon/(2*Math.PI)+.5+1)%1)*image.width,v=(.5-lat/Math.PI)*image.height;const i=(Math.min(image.height-1,Math.floor(v))*image.width+Math.floor(u))*4,o=(y*size+x)*4;const light=.32+.85*Math.max(0,-nx*.5-ny*.25+z*.65);out.data[o]=pixels[i]*light;out.data[o+1]=pixels[i+1]*light;out.data[o+2]=pixels[i+2]*light;out.data[o+3]=Math.min(255,(1-r2)*35000);}
  c.putImageData(out,0,0);earth=sphere;draw();};
 function resize(){const rect=canvas.getBoundingClientRect();width=rect.width;height=rect.height;const dpr=Math.min(devicePixelRatio||1,2);canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);draw();}
 const orbits=[{inc:1.08,angle:.38,r:1.4,color:'#ef9266'},{inc:.48,angle:-.48,r:1.6,color:'#c3f389'},{inc:1.48,angle:-.4,r:1.22,color:'#6fbbdb'}];
 function project(t,orbit,r,cx,cy){let x=Math.cos(t)*r*orbit.r,y=Math.sin(t)*r*orbit.r*Math.cos(orbit.inc),z=Math.sin(t)*r*orbit.r*Math.sin(orbit.inc);const a=orbit.angle;return{x:cx+x*Math.cos(a)-y*Math.sin(a),y:cy+x*Math.sin(a)+y*Math.cos(a),z};}
 function draw(){if(!width||!height)return;ctx.clearRect(0,0,width,height);const cx=width*.52,cy=height*.49,r=Math.min(width*.31,height*.35);
  for(let k=0;k<100;k++){const sx=((k*179.39+37)%997)/997*width,sy=((k*97.71+83)%991)/991*height;ctx.fillStyle=k%7===0?'#9eb9cf70':'#64899e38';ctx.fillRect(sx,sy,k%7===0?1.4:.7,k%7===0?1.4:.7);}
  ctx.save();ctx.strokeStyle='#6e9eb312';ctx.lineWidth=.5;for(let radius of [r*1.15,r*1.85]){ctx.beginPath();ctx.arc(cx,cy,radius,0,2*Math.PI);ctx.stroke();}ctx.beginPath();ctx.moveTo(cx-r*1.95,cy);ctx.lineTo(cx+r*1.95,cy);ctx.moveTo(cx,cy-r*1.7);ctx.lineTo(cx,cy+r*1.7);ctx.stroke();ctx.restore();
  function paths(front){for(const orbit of orbits){ctx.strokeStyle=orbit.color;ctx.globalAlpha=front?.7:.28;ctx.setLineDash(front?[]:[3,5]);ctx.lineWidth=front?1:.7;let active=false;ctx.beginPath();for(let k=0;k<=240;k++){const p=project(k/240*2*Math.PI,orbit,r,cx,cy);if((p.z>=0)===front){if(!active)ctx.moveTo(p.x,p.y);else ctx.lineTo(p.x,p.y);active=true;}else active=false;}ctx.stroke();}ctx.setLineDash([]);ctx.globalAlpha=1;}
  paths(false);if(earth){ctx.drawImage(earth,cx-r,cy-r,2*r,2*r);}else{ctx.fillStyle='#435d4c';ctx.beginPath();ctx.arc(cx,cy,r,0,2*Math.PI);ctx.fill();}
  ctx.strokeStyle='#88cfee70';ctx.beginPath();ctx.arc(cx,cy,r+1,0,2*Math.PI);ctx.stroke();paths(true);
  orbits.forEach((orbit,index)=>{const p=project(phase*(1-index*.15)+index*2.3,orbit,r,cx,cy);if(p.z<0&&Math.hypot(p.x-cx,p.y-cy)<r)return;ctx.save();ctx.translate(p.x,p.y);ctx.rotate(-.25);ctx.fillStyle=orbit.color;ctx.fillRect(-10,-2.5,7,5);ctx.fillRect(4,-2.5,7,5);ctx.fillStyle='#f5ead0';ctx.fillRect(-2,-4,5,8);ctx.restore();ctx.fillStyle='#cdd8c7';ctx.font='8px monospace';ctx.fillText(['OBS-01','RELAY-02','SAT-03'][index],p.x+14,p.y-9);});
  ctx.fillStyle='#8ca38f';ctx.font='8px monospace';ctx.fillText('+ Z / GCRS',width*.5-24,18);
 }
 function animate(time){frame=0;if(last)phase+=Math.min((time-last)/1000,.08)*.13;last=time;draw();schedule();}
 function schedule(){if(running&&visible&&!document.hidden&&!frame)frame=requestAnimationFrame(animate);else if(!running)last=0;}
 new ResizeObserver(resize).observe(canvas);
 if('IntersectionObserver' in window)new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;last=0;schedule();}).observe(canvas);
 document.addEventListener('visibilitychange',()=>{last=0;schedule();});
 reduced.addEventListener('change',()=>{running=!reduced.matches;updateToggle();schedule();});resize();schedule();
}

const preview=document.getElementById('product-preview');
document.querySelector('[data-open-preview]')?.addEventListener('click',()=>preview.showModal());
document.querySelector('[data-close-preview]')?.addEventListener('click',()=>preview.close());
preview?.addEventListener('click',event=>{if(event.target===preview){const rect=preview.getBoundingClientRect();if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom)preview.close();}});
