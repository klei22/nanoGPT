#!/usr/bin/env python3
"""Sample mel multicontext continuations, reconstruct audio, and write Plotly HTML."""
from __future__ import annotations
import argparse, base64, csv, html, json, subprocess, sys
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Run: label:str; csv_path:Path; wav_path:Path; top_k:int; seed:int

def read_rows(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        r=csv.reader(f); header=next(r); rows=[[int(float(x)) for x in row] for row in r if row]
    return header, rows

def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)

def b64(path): return base64.b64encode(Path(path).read_bytes()).decode('ascii')

def wav(repo, csv_path, out, scale, iters):
    subprocess.run([sys.executable, str(repo/'data/audio_mel_csv/mel_int_csv_to_wav.py'), str(csv_path), '--output', str(out), '--scale', str(scale), '--griffin_lim_iters', str(iters)], check=True, cwd=repo)

def sample(repo, ckpt, prompt_csv, output_csv, datasets, holdout, top_k, seed, device, dtype, compile_model):
    cmd=[sys.executable,'sample.py','--out_dir',str(ckpt),'--device',device,'--dtype',dtype,'--multicontext','--multicontext_datasets',*datasets,'--multicontext_csv_input',str(prompt_csv),'--multicontext_csv_output_file',str(output_csv),'--max_new_tokens',str(holdout),'--top_k',str(top_k),'--seed',str(seed),'--num_samples','1','--no-print_model_info']
    cmd.append('--compile' if compile_model else '--no-compile')
    subprocess.run(cmd, check=True, cwd=repo)

def html_page(path, payload):
    js=json.dumps(payload).replace('</','<\\/')
    path.write_text(f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(payload['title'])}</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><style>body{{margin:0;padding:20px;background:#101114;color:#eee;font-family:system-ui,sans-serif}}.panel{{background:#191b20;border:1px solid #333842;border-radius:10px;padding:14px;margin:14px 0}}audio{{width:100%;max-width:720px}}.plot{{height:520px;border:1px solid #343a46;border-radius:8px}}label{{margin-right:14px;color:#cdd1d7}}</style></head><body><h1>{html.escape(payload['title'])}</h1><div class="panel"><p>Audio includes the start/prompt tokens. Ground truth is prompt + held-out continuation; samples are prompt + model continuation.</p><div id="audios"></div></div><div class="panel"><label><input id="truth" type="checkbox" checked> ground truth</label><span id="toggles"></span><div id="heat" class="plot"></div></div><script id="payload" type="application/json">{js}</script><script>
const data=JSON.parse(document.getElementById('payload').textContent); const colors=['Viridis','Magma','Cividis','Turbo'];
const audios=document.getElementById('audios');
function addAudio(name,wav){{const d=document.createElement('div');d.innerHTML=`<h3>${{name}}</h3><audio controls src="data:audio/wav;base64,${{wav}}"></audio>`;audios.appendChild(d)}}
addAudio('ground truth (prompt + holdout)', data.truth.wav); data.samples.forEach(s=>addAudio(s.label, s.wav));
const toggles=document.getElementById('toggles'); data.samples.forEach((s,i)=>{{const l=document.createElement('label');l.innerHTML=`<input type="checkbox" class="s" data-i="${{i}}" checked> ${{s.label}}`;toggles.appendChild(l)}});
document.getElementById('truth').onchange=render; toggles.onchange=render;
function heat(name,z,colorscale,opacity=1){{return {{type:'heatmap',name,z:z[0].map((_,c)=>z.map(r=>r[c])), colorscale, opacity, showscale:false, hovertemplate:'mel=%{{y}}<br>frame=%{{x}}<br>value=%{{z}}<extra>'+name+'</extra>'}}}}
function render(){{const traces=[]; if(document.getElementById('truth').checked) traces.push(heat('ground truth',data.truth.rows,'Viridis',1)); document.querySelectorAll('.s').forEach(cb=>{{if(cb.checked){{const i=+cb.dataset.i; traces.push(heat(data.samples[i].label,data.samples[i].rows,colors[(i+1)%colors.length],0.55));}}}}); Plotly.newPlot('heat',traces,{{paper_bgcolor:'#08090b',plot_bgcolor:'#08090b',font:{{color:'#d6deeb'}},xaxis:{{title:'frame (includes prompt/start tokens)'}},yaxis:{{title:'mel channel'}},margin:{{l:60,r:20,t:20,b:50}}}},{{responsive:true,displaylogo:false}})}}
render();</script></body></html>''', encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--input_csv', default='data/audio_mel_csv/work/mel_int.csv'); p.add_argument('--manifest', default='data/audio_mel_csv/manifest.json'); p.add_argument('--checkpoint_dir', default='out/audio_mel_csv'); p.add_argument('--work_dir', default='out/audio_mel_csv/audio_viewer'); p.add_argument('--holdout_rows', type=int, default=128); p.add_argument('--prompt_rows', type=int, default=512); p.add_argument('--seeds', nargs='+', type=int, default=[1337]); p.add_argument('--top_k', nargs='+', type=int, default=[5]); p.add_argument('--device', default='cuda:0'); p.add_argument('--dtype', default='bfloat16'); p.add_argument('--compile', dest='compile_model', action=argparse.BooleanOptionalAction, default=True); p.add_argument('--skip_sampling', action='store_true'); p.add_argument('--griffin_lim_iters', type=int, default=32)
    a=p.parse_args(); repo=Path(__file__).resolve().parents[2]; work=Path(a.work_dir); work=work if work.is_absolute() else repo/work; work.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((repo/a.manifest if not Path(a.manifest).is_absolute() else Path(a.manifest)).read_text()); scale=manifest.get('quantization',{}).get('scale',32767); datasets=manifest['multicontext_datasets']
    header, rows=read_rows(repo/a.input_csv if not Path(a.input_csv).is_absolute() else Path(a.input_csv)); prompt=rows[:-a.holdout_rows][-a.prompt_rows:]; truth=rows[-a.holdout_rows:]; prompt_csv=work/'prompt.csv'; truth_csv=work/'ground_truth_with_prompt.csv'; write_csv(prompt_csv,header,prompt); write_csv(truth_csv,header,prompt+truth); truth_wav=work/'ground_truth_with_prompt.wav'; wav(repo, truth_csv, truth_wav, scale, a.griffin_lim_iters)
    runs=[]; samples_dir=work/'samples';
    for top_k in a.top_k:
      for seed in a.seeds:
        out=samples_dir/f'sample_topk{top_k}_seed{seed}_with_prompt.csv'; ww=out.with_suffix('.wav')
        if not a.skip_sampling: sample(repo, Path(a.checkpoint_dir), prompt_csv, out, datasets, a.holdout_rows, top_k, seed, a.device, a.dtype, a.compile_model)
        if out.exists(): wav(repo,out,ww,scale,a.griffin_lim_iters); runs.append(Run(f'top_k={top_k} seed={seed}',out,ww,top_k,seed))
    payload={'title':'Mel audio prediction vs ground truth','truth':{'rows':prompt+truth,'wav':b64(truth_wav)},'samples':[{'label':r.label,'rows':read_rows(r.csv_path)[1],'wav':b64(r.wav_path)} for r in runs]}
    out_html=work/'audio_prediction_vs_truth.html'; html_page(out_html,payload); print(f'Viewer HTML: {out_html}')
if __name__=='__main__': main()
