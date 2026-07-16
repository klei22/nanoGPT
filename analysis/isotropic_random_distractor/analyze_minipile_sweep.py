#!/usr/bin/env python3
"""Analyze minipile training sweeps through the isotropic random-distractor lens.

Inputs are the YAML logs produced by optimization_and_search/run_from_yaml.py and,
optionally, the run output directories/checkpoints referenced by those logs. The
analysis is intentionally diagnostic: it checks whether trained minipile models
show width/depth trends and unembedding geometry that are compatible with the toy
model's assumptions; it does not claim the toy model explains the full loss.
"""
from __future__ import annotations

import argparse, csv, html, json, math, os, random, statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def linfit(xs: List[float], ys: List[float]) -> Tuple[float, float, float]:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan"), float("nan"), float("nan")
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my, 0.0, float("nan")
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot else 1.0
    return intercept, slope, r2


def loglog_slope(xs: List[float], ys: List[float]) -> Tuple[float, float, float]:
    pairs = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y > 0 and math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan"), float("nan"), float("nan")
    return linfit([p[0] for p in pairs], [p[1] for p in pairs])


def load_yaml_log(path: Path) -> List[Dict[str, Any]]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read exploration logs")
    docs: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for doc in yaml.safe_load_all(fh):
            if isinstance(doc, dict):
                docs.append(doc)
    return docs


def load_eval_loss(out_dir: Path) -> Dict[str, Any]:
    p = out_dir / "eval_loss.txt"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def row_from_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    cfg = doc.get("config", {}) if isinstance(doc.get("config"), dict) else {}
    out_dir = Path(str(cfg.get("out_dir", doc.get("formatted_name", ""))))
    eval_loss = load_eval_loss(out_dir)
    n_embd = safe_float(cfg.get("n_embd"))
    n_layer = safe_float(cfg.get("n_layer"))
    n_head = safe_float(cfg.get("n_head"))
    num_params = safe_float(doc.get("num_params"))
    if not math.isfinite(num_params) and math.isfinite(n_embd) and math.isfinite(n_layer):
        num_params = 12 * n_layer * n_embd * n_embd
    val_loss = safe_float(eval_loss.get("val", doc.get("best_val_loss")))
    return {
        "name": doc.get("formatted_name", out_dir.name),
        "out_dir": str(out_dir),
        "dataset": cfg.get("dataset", "minipile"),
        "n_embd": n_embd,
        "n_layer": n_layer,
        "n_head": n_head,
        "block_size": safe_float(cfg.get("block_size")),
        "max_iters": safe_float(cfg.get("max_iters")),
        "best_val_loss": val_loss,
        "best_val_bits_per_byte": safe_float(doc.get("best_val_bits_per_byte")),
        "best_val_iter": safe_float(doc.get("best_val_iter")),
        "num_params": num_params,
        "tokens": safe_float(doc.get("best_tokens")),
    }


def checkpoint_geometry(out_dir: Path, pairs: int, seed: int) -> Dict[str, Any]:
    ckpt_path = out_dir / "ckpt.pt"
    if not ckpt_path.exists():
        return {"checkpoint_found": False}
    try:
        import torch  # type: ignore
    except Exception as exc:
        return {"checkpoint_found": True, "geometry_error": f"torch unavailable: {exc}"}
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model", ckpt)
        weight = None
        for key in ("lm_head.weight", "transformer.wte.weight"):
            if key in state:
                weight = state[key].detach().float()
                break
        if weight is None:
            return {"checkpoint_found": True, "geometry_error": "no lm_head.weight or transformer.wte.weight"}
        # Row-normalized unembedding/readout geometry.
        norms = weight.norm(dim=1)
        w = weight / norms.clamp_min(1e-12).unsqueeze(1)
        vocab, d = w.shape
        rng = random.Random(seed)
        cos2 = []
        cos_abs = []
        for _ in range(min(pairs, vocab * max(1, vocab - 1))):
            i = rng.randrange(vocab)
            j = rng.randrange(vocab - 1)
            if j >= i:
                j += 1
            c = float((w[i] * w[j]).sum().item())
            cos2.append(c * c)
            cos_abs.append(abs(c))
        gram_var = mean(cos2)
        centered = w - w.mean(dim=0, keepdim=True)
        cov_trace = float((centered * centered).sum().item() / max(1, vocab - 1))
        # Participation-ratio effective dimension of readout covariance.
        cov = centered.T @ centered / max(1, vocab - 1)
        cov_frob2 = float((cov * cov).sum().item())
        deff = (cov_trace * cov_trace / cov_frob2) if cov_frob2 > 0 else float("nan")
        return {
            "checkpoint_found": True,
            "vocab_size": int(vocab),
            "d": int(d),
            "row_norm_mean": float(norms.mean().item()),
            "row_norm_cv": float((norms.std() / norms.mean().clamp_min(1e-12)).item()),
            "pair_cos2_mean": gram_var,
            "isotropic_cos2_theory": 1.0 / d,
            "abs_cos_mean": mean(cos_abs),
            "readout_deff_pr": deff,
            "readout_deff_ratio": deff / d if d else float("nan"),
        }
    except Exception as exc:
        return {"checkpoint_found": True, "geometry_error": str(exc)}


def plotly_div(div_id: str, rows: List[Dict[str, Any]], xkey: str, ykeys: List[str], title: str, xlabel: str, ylabel: str, logx=False, logy=False) -> str:
    traces = []
    for yk in ykeys:
        traces.append({
            "x": [r.get(xkey) for r in rows],
            "y": [r.get(yk) for r in rows],
            "type": "scatter",
            "mode": "lines+markers",
            "name": yk,
            "line": {"dash": "dash" if any(t in yk for t in ("fit", "prediction", "theory")) else "solid"},
        })
    layout = {"title": title, "xaxis": {"title": xlabel}, "yaxis": {"title": ylabel}, "legend": {"orientation": "h"}}
    if logx:
        layout["xaxis"]["type"] = "log"
    if logy:
        layout["yaxis"]["type"] = "log"
    return f"<div id='{div_id}' class='plot'></div><script>Plotly.newPlot('{div_id}', {json.dumps(traces)}, {json.dumps(layout)}, {{responsive:true, displaylogo:false}});</script>"


def render_report(rows: List[Dict[str, Any]], fits: Dict[str, Any], geometry_rows: List[Dict[str, Any]]) -> str:
    loss_rows = [dict(r, inv_d=(1.0 / r["n_embd"] if r.get("n_embd") else None)) for r in rows]
    geom = [r for r in geometry_rows if r.get("checkpoint_found") and not r.get("geometry_error")]
    summary = html.escape(json.dumps(fits, indent=2))
    raw = html.escape(json.dumps({"runs": rows, "geometry": geometry_rows}, indent=2)[:80000])
    geom_plot = plotly_div("geom", geom, "d", ["pair_cos2_mean", "isotropic_cos2_theory", "readout_deff_ratio"], "Trained readout geometry", "embedding dimension d", "cos² / d_eff ratio", logx=True, logy=True) if geom else "<p>No checkpoint geometry rows were available. Run training with checkpoint saving enabled, then re-run this analyzer.</p>"
    return f"""<!doctype html><meta charset='utf-8'><title>Minipile Isotropic Sweep Analysis</title>
<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>
<style>body{{font-family:system-ui,Arial,sans-serif;max-width:1100px;margin:32px auto;line-height:1.45}}.plot{{height:430px}}pre{{background:#f8fafc;padding:12px;overflow:auto}}section{{border-top:1px solid #ddd;margin-top:24px;padding-top:20px}}</style>
<h1>Minipile training sweep: isotropic random-distractor diagnostics</h1>
<p>This report analyzes trained minipile sweep outputs. It treats validation loss trends and readout geometry as diagnostics of the toy model assumptions, not as proof that the toy model explains all language-model loss.</p>
<section><h2>1. Width trend</h2><p>How to read: if comparable runs differ mostly in width, residual validation loss should often be approximately linear in 1/d. A weak fit means optimization, depth, data, or architecture effects dominate this sweep.</p>{plotly_div('loss_width', loss_rows, 'inv_d', ['best_val_loss'], 'Validation loss against inverse width', '1 / n_embd', 'validation loss')}</section>
<section><h2>2. Parameter scaling</h2><p>How to read: the slope of log(excess loss) versus log(parameter count) is an empirical alpha. Compare it to the conditional alpha=1/gamma only after deciding the architecture path.</p>{plotly_div('loss_params', rows, 'num_params', ['excess_loss'], 'Excess validation loss against parameter count', 'num_params', 'loss above best run', logx=True, logy=True)}</section>
<section><h2>3. Trained readout geometry</h2><p>How to read: isotropic random directions predict mean pairwise cos² near 1/d. The participation-ratio effective dimension should be close to d only when the readout covariance is not highly anisotropic.</p>{geom_plot}</section>
<section><h2>Fit summary</h2><pre>{summary}</pre></section>
<section><h2>Raw analyzed rows</h2><pre>{raw}</pre></section>"""


def analyze(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    finite = [r for r in rows if math.isfinite(r.get("best_val_loss", float("nan")))]
    if not finite:
        return {"error": "no finite validation losses"}
    best = min(r["best_val_loss"] for r in finite)
    for r in rows:
        r["excess_loss"] = max(0.0, r.get("best_val_loss", best) - best)
    inv_d_x = [1.0 / r["n_embd"] for r in finite if r.get("n_embd") and r["n_embd"] > 0]
    inv_d_y = [r["best_val_loss"] for r in finite if r.get("n_embd") and r["n_embd"] > 0]
    width_intercept, width_slope, width_r2 = linfit(inv_d_x, inv_d_y)
    alpha_intercept, alpha_slope, alpha_r2 = loglog_slope([r["num_params"] for r in finite], [r["excess_loss"] for r in finite])
    return {
        "num_runs": len(rows),
        "num_finite_loss_runs": len(finite),
        "best_val_loss": best,
        "loss_vs_inv_width": {"intercept": width_intercept, "slope": width_slope, "r2": width_r2},
        "excess_loss_vs_params": {"empirical_alpha": -alpha_slope if math.isfinite(alpha_slope) else float("nan"), "r2": alpha_r2},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="YAML log from optimization_and_search/run_from_yaml.py")
    ap.add_argument("--outdir", default="report/isotropic_random_distractor_minipile")
    ap.add_argument("--checkpoint-pairs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260715)
    args = ap.parse_args()

    docs = load_yaml_log(Path(args.log))
    rows = [row_from_doc(d) for d in docs]
    fits = analyze(rows)
    geometry_rows = []
    for row in rows:
        geom = checkpoint_geometry(Path(row["out_dir"]), args.checkpoint_pairs, args.seed)
        geometry_rows.append({"name": row["name"], "out_dir": row["out_dir"], **geom})

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "minipile_isotropic_analysis.json").write_text(json.dumps({"fits": fits, "runs": rows, "geometry": geometry_rows}, indent=2), encoding="utf-8")
    with (out / "minipile_isotropic_analysis.csv").open("w", newline="", encoding="utf-8") as fh:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    (out / "index.html").write_text(render_report(rows, fits, geometry_rows), encoding="utf-8")
    print(f"Wrote {out / 'index.html'}")


if __name__ == "__main__":
    main()
