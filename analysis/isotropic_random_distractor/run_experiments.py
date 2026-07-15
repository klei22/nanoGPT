#!/usr/bin/env python3
"""Monte Carlo and analytic probes for the Isotropic Random-Distractor Model.

The script reads an exploration YAML file, runs a suite of lightweight standard-library
experiments, and writes JSON/CSV plus an HTML report with Plotly graphs and dashed prediction traces.
"""
from __future__ import annotations

import argparse, csv, html, json, math, os, random, statistics, sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def softplus(x: float) -> float:
    if x > 40: return x
    if x < -40: return math.exp(x)
    return math.log1p(math.exp(x))

def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x); return 1/(1+z)
    z = math.exp(x); return z/(1+z)

def p0_from_m(m: float) -> float: return sigmoid(-m)

def baseline_loss(M: int, s: float) -> float: return math.log1p(M * math.exp(-s))

def finite_prediction(d: int, M: int, s: float, g: float) -> float:
    m = s - math.log(M); p0 = p0_from_m(m)
    return softplus(-m) + (g*g/(2*d)) * (p0 - p0*p0/M)

def meanfield_prediction(d: int, M: int, s: float, g: float) -> float:
    m = s - math.log(M)
    return softplus(-m + g*g/(2*d))

def sample_unit_dot(d: int) -> float:
    # Dot with fixed h=e1 by drawing a Gaussian vector and normalizing.
    xs = [random.gauss(0, 1) for _ in range(d)]
    norm = math.sqrt(sum(x*x for x in xs))
    return xs[0] / norm if norm else 0.0

def loss_sample(d: int, M: int, s: float, g: float, logit_noise_var: float = 0.0, anisotropic_k: float = 1.0, hard_negatives: int = 0, hard_shift: float = 0.0) -> Tuple[float, float, float]:
    zs = []
    for j in range(M):
        if anisotropic_k == 1.0:
            z = g * sample_unit_dot(d)
        else:
            # Elliptical Gaussian direction with larger first-coordinate variance.
            x0 = random.gauss(0, math.sqrt(anisotropic_k))
            rest = sum(random.gauss(0,1)**2 for _ in range(max(0, d-1)))
            z = g * x0 / math.sqrt(x0*x0 + rest) if d > 1 else g * (1 if x0 >= 0 else -1)
        if logit_noise_var > 0:
            z += random.gauss(0, math.sqrt(logit_noise_var))
        if j < hard_negatives:
            z += hard_shift
        zs.append(z)
    sm = sum(math.exp(z) for z in zs)
    return math.log1p(math.exp(-s) * sm), sm, max(zs) if zs else 0.0

def mean(vals: List[float]) -> float: return sum(vals)/len(vals) if vals else float('nan')

def stderror(vals: List[float]) -> float:
    return statistics.stdev(vals)/math.sqrt(len(vals)) if len(vals) > 1 else 0.0

def linfit(xs: List[float], ys: List[float]) -> Tuple[float,float,float]:
    n=len(xs); mx=mean(xs); my=mean(ys)
    sxx=sum((x-mx)**2 for x in xs); sxy=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    b=sxy/sxx if sxx else 0.0; a=my-b*mx
    ss_tot=sum((y-my)**2 for y in ys); ss_res=sum((y-(a+b*x))**2 for x,y in zip(xs,ys))
    r2=1-ss_res/ss_tot if ss_tot else 1.0
    return a,b,r2

def loglog_slope(xs: List[float], ys: List[float]) -> float:
    pairs=[(math.log(x), math.log(y)) for x,y in zip(xs,ys) if x>0 and y>0]
    return linfit([p[0] for p in pairs],[p[1] for p in pairs])[1] if len(pairs)>1 else float('nan')


def run(cfg: Dict[str, Any]) -> Dict[str, Any]:
    random.seed(int(cfg.get('seed', 1234)))
    base = cfg.get('base', {})
    M=int(base.get('M', 64)); m=float(base.get('m', 0.0)); s=float(base.get('s', math.log(M)+m)); g=float(base.get('g', 2.0))
    dims=[int(x) for x in cfg.get('dimensions', [8,16,32,64,128,256])]
    samples=int(cfg.get('samples_per_point', 400))
    out: Dict[str, Any]={'config':cfg, 'facets':{}}

    # 1 spherical moments and loss law
    rows=[]
    for d in dims:
        dots=[sample_unit_dot(d) for _ in range(samples)]
        losses=[]; sms=[]; maxes=[]
        for _ in range(samples):
            l,smv,mx=loss_sample(d,M,s,g); losses.append(l); sms.append(smv); maxes.append(mx)
        rows.append({'d':d,'inv_d':1/d,'E_u':mean(dots),'E_u2':mean([u*u for u in dots]),'theory_u2':1/d,
          'E_u4':mean([u**4 for u in dots]),'theory_u4':3/(d*(d+2)), 'loss_mc':mean(losses), 'loss_se':stderror(losses),
          'loss_finite':finite_prediction(d,M,s,g), 'loss_meanfield':meanfield_prediction(d,M,s,g), 'baseline':baseline_loss(M,s),
          'partition_cv2_mc': (statistics.pvariance(sms)/(mean(sms)**2)) if len(sms)>1 else 0.0,
          'partition_cv2_gaussian': (math.exp(g*g/d)-1)/M, 'max_logit_mean':mean(maxes)})
    a,b,r2=linfit([r['inv_d'] for r in rows],[r['loss_mc']-r['baseline'] for r in rows])
    C=(g*g/2)*(p0_from_m(s-math.log(M))-p0_from_m(s-math.log(M))**2/M)
    out['facets']['finite_vocabulary']={'rows':rows,'fit_intercept':a,'fit_slope':b,'fit_r2':r2,'theory_slope':C,'pass':abs(b-C)/max(abs(C),1e-9)<0.25 and r2>0.9}

    # 2 margin regimes
    margins=[float(x) for x in cfg.get('margin_sweep', [-4,-2,0,2,4])]
    mr=[]
    d0=max(dims)
    for mm in margins:
        ss=math.log(M)+mm
        losses=[loss_sample(d0,M,ss,g)[0] for _ in range(samples)]
        p=p0_from_m(mm); pen=mean(losses)-baseline_loss(M,ss)
        mr.append({'m':mm,'P0':p,'loss_penalty_mc':pen,'loss_penalty_theory':(g*g/(2*d0))*(p-p*p/M)})
    out['facets']['margin_contexts']={'rows':mr}

    # 3 anisotropy/effective dimension and hard negatives
    ar=[]
    for k in [float(x) for x in cfg.get('anisotropy_k', [1,2,4,8])]:
        d=d0; deff=(k + d - 1)/k
        losses=[loss_sample(d,M,s,g,anisotropic_k=k)[0] for _ in range(samples)]
        ar.append({'anisotropy_k':k,'d_eff':deff,'loss_mc':mean(losses),'prediction_d_eff':baseline_loss(M,s)+(g*g/(2*deff))*(p0_from_m(m)-p0_from_m(m)**2/M)})
    hn=[]
    for hcnt,shift in cfg.get('hard_negative_cases', [[0,0.0],[1,1.5],[4,1.5]]):
        losses=[]; shares=[]
        for _ in range(samples):
            l,smv,mx=loss_sample(d0,M,s,g,hard_negatives=int(hcnt),hard_shift=float(shift)); losses.append(l); shares.append(math.exp(mx)/smv)
        hn.append({'hard_negatives':hcnt,'hard_shift':shift,'loss_mc':mean(losses),'max_partition_share':mean(shares),'isotropic_prediction':finite_prediction(d0,M,s,g)})
    out['facets']['anisotropy_and_tails']={'anisotropy_rows':ar,'hard_negative_rows':hn}

    # 4 logit scale, quantization noise
    gr=[]
    for r in [float(x) for x in cfg.get('g_scaling_exponents', [0,0.25,0.5,0.75])]:
        vals=[]
        for d in dims:
            gd=g*(d/dims[0])**r
            losses=[loss_sample(d,M,s,gd)[0] for _ in range(max(80, samples//3))]
            vals.append({'d':d,'g_d':gd,'penalty':mean(losses)-baseline_loss(M,s),'variance_scale':gd*gd/d})
        gr.append({'r':r,'rows':vals,'penalty_loglog_slope':loglog_slope([v['d'] for v in vals],[v['penalty'] for v in vals])})
    qr=[]
    for qv in [float(x) for x in cfg.get('noise_variances', [0,0.005,0.01,0.02,0.04])]:
        losses=[loss_sample(d0,M,s,g,logit_noise_var=qv)[0] for _ in range(samples)]
        qr.append({'noise_var':qv,'loss_mc':mean(losses),'local_prediction':baseline_loss(M,s)+0.5*(p0_from_m(m)-p0_from_m(m)**2/M)*(g*g/d0+qv)})
    out['facets']['scale_and_noise']={'g_scaling':gr,'quantization_noise':qr}

    # 5 parameter and compute laws are algebraic checks
    gammas=[float(x) for x in cfg.get('architecture_gammas', [1,2,3])]
    pr=[]
    for gamma in gammas:
        Ns=[(d**gamma) for d in dims]
        losses=[baseline_loss(M,s)+C/(N**(1/gamma)) for N in Ns]
        pr.append({'gamma':gamma,'expected_alpha':1/gamma,'fit_alpha':-loglog_slope(Ns,[x-baseline_loss(M,s) for x in losses])})
    beta=float(cfg.get('data_beta', 1/3)); alpha=float(cfg.get('model_alpha', 1/3))
    out['facets']['scaling_laws']={'parameter_rows':pr,'compute_optimal':{'alpha':alpha,'beta':beta,'N_star_exponent':beta/(alpha+beta),'D_star_exponent':alpha/(alpha+beta),'loss_exponent':alpha*beta/(alpha+beta)}}
    return out


def plotly_div(div_id: str, series: List[Dict[str,Any]], xkey: str, ykeys: List[str], title: str, xlabel: str, ylabel: str, logx=False, logy=False) -> str:
    traces=[]
    for yk in ykeys:
        traces.append({
            'x':[row[xkey] for row in series],
            'y':[row[yk] for row in series],
            'type':'scatter', 'mode':'lines+markers', 'name':yk,
            'line': {'dash': 'dash' if any(tok in yk for tok in ['prediction','theory','finite','meanfield']) else 'solid'}
        })
    layout={'title':title,'xaxis':{'title':xlabel},'yaxis':{'title':ylabel},'legend':{'orientation':'h'},'margin':{'t':60,'l':70,'r':30,'b':60}}
    if logx: layout['xaxis']['type']='log'
    if logy: layout['yaxis']['type']='log'
    return f"<div id='{div_id}' class='plot'></div><script>Plotly.newPlot('{div_id}', {json.dumps(traces)}, {json.dumps(layout)}, {{responsive:true, displaylogo:false}});</script>"

def html_report(res: Dict[str, Any]) -> str:
    f=res['facets']; fv=f['finite_vocabulary']; rows=fv['rows']
    verdict='PASS' if fv['pass'] else 'CHECK'
    body=f"""<!doctype html><meta charset='utf-8'><title>Isotropic Random-Distractor Report</title>
<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>
<style>body{{font-family:system-ui,Arial,sans-serif;max-width:1100px;margin:32px auto;line-height:1.45}}code,pre{{background:#f4f4f5;padding:2px 4px}}section{{border-top:1px solid #ddd;padding-top:24px;margin-top:24px}}.card{{background:#f8fafc;border:1px solid #e2e8f0;padding:12px;border-radius:10px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:4px 7px}}.plot{{width:100%;height:430px}}</style>
<h1>Isotropic Random-Distractor Model: Demo Report</h1>
<p class='card'><b>Overall finite-vocabulary test:</b> {verdict}. Fitted penalty slope versus 1/d is {fv['fit_slope']:.4g}; theory predicts {fv['theory_slope']:.4g}; R²={fv['fit_r2']:.4f}.</p>
<section><h2>1. Spherical moments and finite-vocabulary loss law</h2><p>How to read: solid points/lines are Monte Carlo estimates; dashed lines are analytic predictions. The x-axis is 1/d, so a straight line supports the inverse-dimension term. The mean-field curve should usually sit above the exact finite-M prediction because it replaces a random partition sum by its mean.</p>{plotly_div('loss_inv_d', rows,'inv_d',['loss_mc','loss_finite','loss_meanfield'],'Expected loss vs inverse dimension','1/d','loss')}</section>
<section><h2>2. Angular moment checks</h2><p>How to read: dashed theory curves show E[u²]=1/d and E[u⁴]=3/(d(d+2)). Agreement checks the isotropic sphere sampler before testing softmax claims.</p>{plotly_div('moments', rows,'d',['E_u2','theory_u2','E_u4','theory_u4'],'Spherical moment identities','d','moment',logx=True,logy=True)}</section>
<section><h2>3. Partition concentration</h2><p>How to read: lower CV² means the random partition sum is concentrated and the mean-field closure is safer. Dashed theory is the Gaussian closure value (exp(g²/d)-1)/M.</p>{plotly_div('partition', rows,'d',['partition_cv2_mc','partition_cv2_gaussian'],'Partition-sum concentration','d','CV²',logx=True,logy=True)}</section>
<section><h2>4. Context margin regimes</h2><p>How to read: the penalty is small for easy contexts (large positive m, small P₀) and large for hard contexts (negative m, P₀≈1). Dashed line is the local finite-vocabulary prediction.</p>{plotly_div('margins', f['margin_contexts']['rows'],'m',['loss_penalty_mc','loss_penalty_theory'],'Margin-dependent variance penalty','m = s - log M','loss penalty')}</section>
<section><h2>5. Anisotropy and hard-negative tails</h2><p>How to read: anisotropy is summarized by d_eff; when the effective-dimension correction works, the Monte Carlo loss follows the dashed d_eff prediction. Hard negatives show a failure mode: a small shifted cluster can dominate the partition, making the isotropic prediction over-optimistic.</p>{plotly_div('anisotropy', f['anisotropy_and_tails']['anisotropy_rows'],'d_eff',['loss_mc','prediction_d_eff'],'Anisotropy through effective dimension','d_eff','loss')}{plotly_div('hard_negatives', f['anisotropy_and_tails']['hard_negative_rows'],'hard_negatives',['loss_mc','isotropic_prediction','max_partition_share'],'Hard-negative tail stress test','number of shifted negatives','loss / max share')}</section>
<section><h2>6. Logit-scale growth and added logit noise</h2><p>How to read: if g grows as d^r, the penalty should scale roughly like d^(2r-1) while the small-variance approximation remains valid. For added zero-mean noise, degradation should be smooth and near-linear in the added variance.</p>{plotly_div('noise', f['scale_and_noise']['quantization_noise'],'noise_var',['loss_mc','local_prediction'],'Independent logit-noise degradation','added noise variance','loss')}</section>
<section><h2>7. Conditional parameter and compute scaling</h2><p>How to read: these are algebraic implications, not Monte Carlo discoveries. The exponent alpha equals 1/gamma only after choosing an architecture path N∝d^gamma. Compute-optimal exponents require an independently supplied data exponent beta.</p><pre>{html.escape(json.dumps(f['scaling_laws'],indent=2))}</pre></section>
<section><h2>Raw results</h2><pre>{html.escape(json.dumps(res,indent=2)[:60000])}</pre></section>"""
    return body


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--config', default='explorations/isotropic_random_distractor.yaml'); ap.add_argument('--outdir', default='report/isotropic_random_distractor'); ap.add_argument('--samples', type=int)
    args=ap.parse_args()
    text=Path(args.config).read_text()
    cfg=yaml.safe_load(text) if yaml else json.loads(text)
    if args.samples: cfg['samples_per_point']=args.samples
    res=run(cfg)
    out=Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    (out/'results.json').write_text(json.dumps(res,indent=2))
    (out/'index.html').write_text(html_report(res))
    # flatten key CSVs
    for name in ['finite_vocabulary','margin_contexts']:
        rows=res['facets'][name]['rows']
        with (out/f'{name}.csv').open('w',newline='') as fh:
            w=csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"Wrote {out/'index.html'}")

if __name__ == '__main__': main()
