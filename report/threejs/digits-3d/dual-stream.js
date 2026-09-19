// This viewer reads exported data only. Training stays in the existing Python GPT.
const $ = (id) => document.getElementById(id);
const variantNames = {table_free:'Free lookup table', table_sphere:'Spherical lookup table',
  great_circle:'Great circle', small_circle:'Learned small circle'};
const query = new URLSearchParams(location.search);
const manifestURL = new URL(query.get('manifest') || 'dual-stream/manifest.json', location.href);
const state = {runs:[], selected:[null,null], views:[], iteration:0, times:[], playing:false, epoch:0};
const colors = [0xffba66, 0x6ed3de];
let THREE, OrbitControls;

function showError(error) {
  $('error').hidden=false;
  $('error').textContent = `${error.message}\nTrain with: TASK_MODE=dual_stream bash demos/digits_3d_trajectory_demo.sh\nThen serve the repository with: python3 -m http.server 8000`;
}
async function readJSON(url) {
  const response=await fetch(url);
  if(!response.ok) throw new Error(`Could not load ${url}: HTTP ${response.status}`);
  return response.json();
}
function frameAt(data, iteration) {
  let lo=0, hi=data.frames.length-1;
  while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(data.frames[mid].iteration<=iteration)lo=mid;else hi=mid-1;}
  return {frame:data.frames[lo], index:lo};
}
function circlePoint(g,t) {
  return new THREE.Vector3(...g.center).addScaledVector(new THREE.Vector3(...g.u),g.radius*Math.cos(2*Math.PI*t))
    .addScaledVector(new THREE.Vector3(...g.v),g.radius*Math.sin(2*Math.PI*t));
}
function textSprite(text,color,opacity) {
  const canvas=document.createElement('canvas');canvas.width=128;canvas.height=128;
  const ctx=canvas.getContext('2d');ctx.font='500 76px system-ui';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillStyle='#ffffff';ctx.fillText(text,64,64);
  const texture=new THREE.CanvasTexture(canvas);
  const sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:texture,color,transparent:true,depthTest:false,opacity}));
  sprite.scale.set(.30,.30,1);return sprite;
}
function makeLine(color,opacity=1) {
  return new THREE.Line(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color,transparent:true,opacity}));
}
function setLine(line,points) {
  line.geometry.dispose();line.geometry=new THREE.BufferGeometry().setFromPoints(points);
}
function disposeGroup(group) {
  group.traverse(o=>{o.geometry?.dispose();if(o.material){o.material.map?.dispose();o.material.dispose();}});
  group.clear();
}
class GeometryView {
  constructor(side) {
    this.side=side;this.canvas=$(`${side}-scene`);
    this.renderer=new THREE.WebGLRenderer({canvas:this.canvas,antialias:true});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio,2));
    this.renderer.outputColorSpace=THREE.SRGBColorSpace;
    this.scene=new THREE.Scene();this.scene.background=new THREE.Color(0x101722);
    this.camera=new THREE.PerspectiveCamera(43,1,.01,200);
    this.controls=new OrbitControls(this.camera,this.canvas);this.controls.enableDamping=true;
    this.scene.add(new THREE.AmbientLight(0xffffff,2));
    this.group=new THREE.Group();this.scene.add(this.group);
    new ResizeObserver(()=>this.resize()).observe(this.canvas.parentElement);
  }
  resize() {
    const width=this.canvas.clientWidth, height=this.canvas.clientHeight;
    if(!width||!height)return;
    this.renderer.setSize(width,height,false);this.camera.aspect=width/height;this.camera.updateProjectionMatrix();
  }
  load(data) {
    this.data=data;disposeGroup(this.group);this.tokens=[];this.curves=[];this.centers=[];this.virtual=[];
    let extent=data.config.radius;
    for(const frame of data.frames)for(const p of frame.positions)extent=Math.max(extent,Math.hypot(...p));
    this.extent=extent;this.camera.position.set(extent*2,extent*1.3,extent*2.4);
    this.controls.target.set(0,0,0);this.controls.update();
    const shell=new THREE.Mesh(new THREE.SphereGeometry(data.config.radius,28,18),
      new THREE.MeshBasicMaterial({color:0x72879c,wireframe:true,transparent:true,opacity:.12}));
    // For the free table this is a reference radius, not a constraint.
    this.group.add(shell);
    this.group.add(new THREE.AxesHelper(data.config.radius*.35));
    for(let i=0;i<data.tokens.length;i++){
      const group=data.groups.find(g=>i>=g.start&&i<g.start+g.size), gi=data.groups.indexOf(group);
      const active=i-group.start<group.active;
      const dot=new THREE.Mesh(new THREE.SphereGeometry(.045,12,8),
        new THREE.MeshBasicMaterial({color:colors[gi],wireframe:!active,transparent:true,opacity:active?1:.35}));
      const label=textSprite(data.tokens[i],colors[gi],active?1:.4), trail=makeLine(colors[gi],active?.5:.15);
      this.group.add(dot,label,trail);this.tokens.push({dot,label,trail});
    }
    for(let i=0;i<2;i++){
      const line=makeLine(colors[i],.8);
      const center=new THREE.Mesh(new THREE.BoxGeometry(.05,.05,.05),new THREE.MeshBasicMaterial({color:colors[i]}));
      const virtual=new THREE.Mesh(new THREE.OctahedronGeometry(.075),new THREE.MeshBasicMaterial({color:colors[i],wireframe:true}));
      this.group.add(line,center,virtual);this.curves.push(line);this.centers.push(center);this.virtual.push(virtual);
    }
    this.sums=new THREE.Points(new THREE.BufferGeometry(),new THREE.PointsMaterial({color:0xc9d2db,size:.026,transparent:true,opacity:.55}));
    this.group.add(this.sums);this.resize();
  }
  renderFrame(iteration) {
    if(!this.data)return;
    const data=this.data, {frame,index}=frameAt(data,iteration);
    this.tokens.forEach((object,i)=>{
      const p=new THREE.Vector3(...frame.positions[i]);object.dot.position.copy(p);
      object.label.position.copy(p).add(new THREE.Vector3(0,.11,0));
      object.trail.visible=$('trails').checked;
      if(object.trail.visible)setLine(object.trail,data.frames.slice(Math.max(0,index-99),index+1).map(f=>new THREE.Vector3(...f.positions[i])));
    });
    for(let i=0;i<2;i++){
      const geometry=frame.circles[i], visible=Boolean(geometry);
      this.curves[i].visible=visible&&$('circles').checked;this.centers[i].visible=visible;
      this.virtual[i].visible=visible;
      if(geometry){
        if(this.curves[i].visible)setLine(this.curves[i],Array.from({length:97},(_,j)=>circlePoint(geometry,j/96)));
        this.centers[i].position.fromArray(geometry.center);
        this.virtual[i].position.copy(circlePoint(geometry,Number($(i===0?'digit-phase':'letter-phase').value)));
      }
    }
    this.sums.visible=$('sums').checked;
    if(this.sums.visible){
      const [d,l]=data.groups, pairs=[];
      if(data.config.pairing==='independent'){
        for(let i=0;i<d.active;i++)for(let j=0;j<l.active;j++)pairs.push([i,j]);
      }else{
        const gcd=(a,b)=>b?gcd(b,a%b):a;
        for(let k=0;k<d.active*l.active/gcd(d.active,l.active);k++)pairs.push([k%d.active,k%l.active]);
      }
      this.sums.geometry.dispose();this.sums.geometry=new THREE.BufferGeometry().setFromPoints(pairs.map(([i,j])=>
        new THREE.Vector3(...frame.positions[d.start+i]).add(new THREE.Vector3(...frame.positions[l.start+j]))));
    }
    const m=frame.metrics, container=$(`${this.side}-metrics`);container.replaceChildren();
    const lines=[`Iteration ${frame.iteration} · ${data.parameter_count} parameters · mean CE ${m.loss.toFixed(4)}`,
      `Digits: CE ${m.digit_loss.toFixed(4)} / ${(100*m.digit_accuracy).toFixed(1)}% · Letters: CE ${m.letter_loss.toFixed(4)} / ${(100*m.letter_accuracy).toFixed(1)}%`,
      `Joint accuracy ${(100*m.joint_accuracy).toFixed(1)}% · vector norms ${m.norm_min.toFixed(3)}–${m.norm_max.toFixed(3)}`];
    if(frame.circles[0])lines.push(`Center offsets: ${frame.circles.map(g=>g.offset.toFixed(3)).join(' / ')} · numeric virtual value ${(Number($('digit-phase').value)*data.config.digit_slots).toFixed(2)}`);
    else lines.push('Free table' + (data.fixed_norm?' directions, projected after each update.':'; sphere shows the initial reference radius.'));
    lines.forEach((text,i)=>{const p=document.createElement('p');const el=document.createElement(i===0?'strong':'span');el.textContent=text;p.appendChild(el);container.appendChild(p);});
    this.canvas.dataset.iteration=String(frame.iteration);
  }
  tick(){this.controls.update();this.renderer.render(this.scene,this.camera);}
}
function drawLoss() {
  if(state.selected.some(d=>!d))return;
  const canvas=$('loss-chart'), width=canvas.clientWidth, height=220, ratio=Math.min(devicePixelRatio,2);
  canvas.width=Math.round(width*ratio);canvas.height=height*ratio;
  const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);
  const margin={left:52,right:15,top:15,bottom:39}, key=$('loss-kind').value;
  const maxStep=Number($('iteration').max)||1;
  let ymax=key==='joint_accuracy'?1:0;
  for(const data of state.selected)for(const f of data.frames)ymax=Math.max(ymax,f.metrics[key]);
  ymax=Math.max(ymax*1.06,1e-6);
  const px=t=>margin.left+t/maxStep*(width-margin.left-margin.right);
  const py=v=>height-margin.bottom-v/ymax*(height-margin.bottom-margin.top);
  ctx.font='12px system-ui';ctx.lineWidth=1;
  for(let j=0;j<=3;j++){
    const value=j*ymax/3,y=py(value);ctx.strokeStyle='#314154';ctx.beginPath();ctx.moveTo(margin.left,y);ctx.lineTo(width-margin.right,y);ctx.stroke();
    ctx.fillStyle='#a9b9ca';ctx.textAlign='right';ctx.fillText(value.toFixed(2),margin.left-8,y+4);
    const t=Math.round(j*maxStep/3);ctx.textAlign=j===0?'left':j===3?'right':'center';ctx.fillText(String(t),px(t),height-margin.bottom+19);
  }
  ctx.textAlign='center';ctx.fillText('Optimizer updates',width/2,height-3);
  ctx.save();ctx.translate(12,height/2-10);ctx.rotate(-Math.PI/2);ctx.fillText(key==='joint_accuracy'?'Accuracy':'Cross-entropy (nats)',0,0);ctx.restore();
  const palette=['#bdacff','#a2dc91'];
  state.selected.forEach((data,s)=>{
    ctx.strokeStyle=palette[s];ctx.lineWidth=2;ctx.beginPath();
    data.frames.forEach((f,i)=>{i?ctx.lineTo(px(f.iteration),py(f.metrics[key])):ctx.moveTo(px(f.iteration),py(f.metrics[key]));});ctx.stroke();
    const {frame}=frameAt(data,state.iteration);ctx.fillStyle=palette[s];ctx.beginPath();ctx.arc(px(frame.iteration),py(frame.metrics[key]),4,0,Math.PI*2);ctx.fill();
  });
  ctx.strokeStyle='#eef3f8';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(px(state.iteration),margin.top);ctx.lineTo(px(state.iteration),height-margin.bottom);ctx.stroke();
  canvas.dataset.metric=key;canvas.dataset.iteration=String(state.iteration);
}
function updateFrame() {
  state.iteration=Number($('iteration').value);
  $('iteration-label').value=String(state.iteration);
  $('digit-phase-label').value=Number($('digit-phase').value).toFixed(3);
  $('letter-phase-label').value=Number($('letter-phase').value).toFixed(3);
  state.views.forEach(v=>v.renderFrame(state.iteration));drawLoss();
}
function stepBy(delta) {
  const current=state.times.findIndex(t=>t>=state.iteration);
  const index=Math.max(0,Math.min(state.times.length-1,Math.max(0,current)+delta));
  $('iteration').value=String(state.times[index]);updateFrame();
}
function togglePlay(){state.playing=!state.playing;$('play').textContent=state.playing?'Pause':'Play';}
async function selectRuns() {
  const epoch=++state.epoch;state.playing=false;$('play').textContent='Play';
  const entries=[$('left-run').value,$('right-run').value].map(name=>state.runs.find(r=>r.name===name));
  const data=await Promise.all(entries.map(run=>readJSON(new URL(run.file,manifestURL))));
  if(epoch!==state.epoch)return;
  for(const run of data){
    if(run.task!=='dual_stream_clock'||!run.frames?.length||run.groups?.length!==2)throw new Error('This viewer requires a dual_stream_clock trajectory.');
  }
  state.selected=data;state.times=[...new Set(data.flatMap(d=>d.frames.map(f=>f.iteration)))].sort((a,b)=>a-b);
  $('iteration').max=String(state.times[state.times.length-1]);
  $('iteration').value=String(Math.min(state.iteration,Number($('iteration').max)));
  state.views.forEach((v,i)=>v.load(data[i]));
  const c=data[0].config;
  const mismatches=['digits','digit_slots','letters','pairing','block_size','batch_size','learning_rate','weight_decay','embedding_init','heads','radius','mlp_expansion'].filter(k=>data[0].config[k]!==data[1].config[k]);
  $('data-description').textContent=mismatches.length?`Different settings: ${mismatches.join(', ')}`:
    `0–${c.digits-1} + a–${String.fromCharCode(96+c.letters)} · ${c.digit_slots} numeric slots · ${c.heads} attention head${c.heads===1?'':'s'}`;
  ['iteration','previous','next','play'].forEach(id=>$(id).disabled=false);
  query.set('left',entries[0].name);query.set('right',entries[1].name);
  history.replaceState(null,'',`${location.pathname}?${query}`);
  updateFrame();
}
async function main() {
  const imports=await Promise.all([import('https://esm.sh/three@0.160.0'),
    import('https://esm.sh/three@0.160.0/examples/jsm/controls/OrbitControls.js'),readJSON(manifestURL)]);
  THREE=imports[0];OrbitControls=imports[1].OrbitControls;state.runs=imports[2].runs||[];
  if(!state.runs.length)throw new Error('No completed runs in this manifest.');
  state.views=['left','right'].map(side=>new GeometryView(side));
  ['left','right'].forEach((side,i)=>{
    const select=$(`${side}-run`);
    for(const run of state.runs){const option=document.createElement('option');option.value=run.name;option.textContent=`${variantNames[run.variant]||run.variant} · seed ${run.seed}`;select.appendChild(option);}
    const preferred=query.get(side)||state.runs.find(r=>r.variant===(i?'small_circle':'table_sphere'))?.name;
    if(state.runs.some(r=>r.name===preferred))select.value=preferred;
    select.disabled=false;select.addEventListener('change',()=>selectRuns().catch(showError));
  });
  $('iteration').addEventListener('input',updateFrame);
  ['digit-phase','letter-phase','circles','trails','sums'].forEach(id=>$(id).addEventListener('input',updateFrame));
  $('loss-kind').addEventListener('change',drawLoss);
  $('previous').addEventListener('click',()=>stepBy(-1));$('next').addEventListener('click',()=>stepBy(1));
  $('play').addEventListener('click',togglePlay);
  document.addEventListener('keydown',event=>{
    if(/INPUT|SELECT|TEXTAREA|BUTTON/.test(event.target.tagName)||$('iteration').disabled)return;
    if(event.code==='Space'){event.preventDefault();togglePlay();}
    if(event.code==='ArrowLeft'||event.code==='ArrowRight'){event.preventDefault();stepBy(event.code==='ArrowRight'?1:-1);}
  });
  $('loss-chart').addEventListener('pointerdown',event=>{
    const rect=event.currentTarget.getBoundingClientRect();
    $('iteration').value=String(Math.round(Math.max(0,Math.min(1,(event.clientX-rect.left-52)/(rect.width-67)))*Number($('iteration').max)));
    updateFrame();
  });
  new ResizeObserver(drawLoss).observe($('loss-chart'));
  await selectRuns();let last=0;
  function tick(now){
    if(state.playing&&now-last>50){if(state.iteration>=Number($('iteration').max)){$('iteration').value='0';updateFrame();}else stepBy(1);last=now;}
    state.views.forEach(v=>v.tick());requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}
main().catch(showError);
