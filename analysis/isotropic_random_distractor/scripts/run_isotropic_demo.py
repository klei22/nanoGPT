#!/usr/bin/env python3
"""Explore the Isotropic Random-Distractor Model and render an HTML report.

The script intentionally depends only on the Python standard library so it can be
run in lightweight nanoGPT checkouts. It accepts a small YAML subset consisting
of ``key: value`` pairs with scalar numbers/strings and one-line lists.
"""
from __future__ import annotations

import argparse
import ast
import csv
import html
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List


def softplus(x: float) -> float:
    if x > 40:
        return x
    if x < -40:
        return math.exp(x)
    return math.log1p(math.exp(x))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def parse_yaml_subset(path: Path) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        try:
            cfg[key.strip()] = ast.literal_eval(value)
        except Exception:
            cfg[key.strip()] = value
    return cfg


def spherical_cosine(d: int, rng: random.Random) -> float:
    # If w is uniform on S^{d-1}, w_1^2 ~ Beta(1/2, (d-1)/2).
    # Sampling this one coordinate directly is much faster than drawing d
    # Gaussians and normalizing for every distractor.
    if d <= 1:
        return 1.0 if rng.random() < 0.5 else -1.0
    magnitude = math.sqrt(rng.betavariate(0.5, 0.5 * (d - 1)))
    return magnitude if rng.random() < 0.5 else -magnitude


def loss_from_logits(s: float, logits: Iterable[float]) -> float:
    # Stable enough because demo logits are modest; avoids external numerical deps.
    return math.log1p(math.exp(-s) * sum(math.exp(z) for z in logits))


def finite_theory(M: int, s: float, g: float, d: int) -> float:
    m = s - math.log(M)
    p0 = sigmoid(-m)
    return softplus(-m) + (g * g / (2.0 * d)) * (p0 - p0 * p0 / M)


def mean_field(M: int, s: float, g: float, d: int) -> float:
    m = s - math.log(M)
    return softplus(-m + g * g / (2.0 * d))


def run_trials(M: int, s: float, g: float, d: int, trials: int, rng: random.Random) -> float:
    total = 0.0
    for _ in range(trials):
        total += loss_from_logits(s, (g * spherical_cosine(d, rng) for _ in range(M)))
    return total / trials


def svg_line_chart(rows: List[Dict[str, float]], xkey: str, series: List[str], title: str, ylabel: str) -> str:
    width, height, pad = 760, 320, 48
    xs = [float(r[xkey]) for r in rows]
    ys = [float(r[k]) for r in rows for k in series]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    if ymax == ymin:
        ymax += 1.0
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]
    def sx(x: float) -> float:
        return pad + (x - xmin) / (xmax - xmin or 1.0) * (width - 2 * pad)
    def sy(y: float) -> float:
        return height - pad - (y - ymin) / (ymax - ymin) * (height - 2 * pad)
    parts = [f'<h3>{html.escape(title)}</h3><svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">']
    parts.append(f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#444"/><line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#444"/>')
    parts.append(f'<text x="{width/2}" y="{height-8}" text-anchor="middle">{html.escape(xkey)}</text><text x="15" y="{height/2}" transform="rotate(-90 15 {height/2})" text-anchor="middle">{html.escape(ylabel)}</text>')
    for i, key in enumerate(series):
        pts = " ".join(f'{sx(float(r[xkey])):.1f},{sy(float(r[key])):.1f}' for r in rows)
        parts.append(f'<polyline fill="none" stroke="{colors[i % len(colors)]}" stroke-width="2.5" points="{pts}"/>')
        parts.append(f'<text x="{width-pad-120}" y="{pad+18*i}" fill="{colors[i % len(colors)]}">{html.escape(key)}</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="analysis/isotropic_random_distractor/configs/base_demo.yaml")
    ap.add_argument("--outdir", default="analysis/isotropic_random_distractor/reports/base_demo")
    args = ap.parse_args()
    cfg = parse_yaml_subset(Path(args.config))
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(cfg.get("seed", 0)))
    M, s, g, trials = int(cfg["M"]), float(cfg["s"]), float(cfg["g"]), int(cfg["trials"])
    dims = [int(x) for x in cfg["dims"]]

    scaling = []
    for d in dims:
        mc = run_trials(M, s, g, d, trials, rng)
        th = finite_theory(M, s, g, d)
        mf = mean_field(M, s, g, d)
        scaling.append({"d": d, "monte_carlo": mc, "finite_theory": th, "mean_field": mf, "abs_error": abs(mc - th), "jensen_gap": mf - mc})
    write_csv(out / "dimension_scaling.csv", scaling)

    margins = []
    for m in [float(x) for x in cfg["margin_values"]]:
        p0 = sigmoid(-m)
        margins.append({"margin_m": m, "baseline_softplus": softplus(-m), "p0": p0, "variance_coefficient": 0.5 * g * g * (p0 - p0 * p0 / M)})
    write_csv(out / "margin_regimes.csv", margins)

    quant = []
    m = s - math.log(M); p0 = sigmoid(-m); d0 = dims[-1]
    for vq in [float(x) for x in cfg["quantization_variances"]]:
        quant.append({"sigma_q2": vq, "predicted_delta_loss": 0.5 * (p0 - p0 * p0 / M) * (g * g / d0 + vq)})
    write_csv(out / "quantization_noise.csv", quant)

    arch = []
    beta = float(cfg.get("data_beta", 1/3))
    for gamma in [float(x) for x in cfg["architecture_gamma_values"]]:
        alpha = 1.0 / gamma
        arch.append({"gamma": gamma, "alpha_model": alpha, "N_star_compute_exponent": beta/(alpha+beta), "D_star_compute_exponent": alpha/(alpha+beta), "loss_compute_exponent": alpha*beta/(alpha+beta)})
    write_csv(out / "architecture_compute.csv", arch)

    verdict = "holds" if max(r["abs_error"] for r in scaling) < 0.03 else "partially holds"
    html_doc = f"""<!doctype html><meta charset='utf-8'><title>Isotropic Random-Distractor Report</title>
<style>body{{font:16px system-ui;margin:2rem;max-width:980px}} svg{{width:100%;height:auto;border:1px solid #ddd}} code{{background:#f4f4f5;padding:0.1rem 0.25rem}} table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:.35rem}}</style>
<h1>Isotropic Random-Distractor Model Exploration</h1>
<p><b>Configuration:</b> <code>{html.escape(json.dumps(cfg))}</code></p>
<h2>Prediction report</h2><p>The finite-vocabulary <code>1/d</code> prediction <b>{verdict}</b> in this run, based on the largest absolute Monte Carlo-vs-theory error in <code>dimension_scaling.csv</code>. The mean-field curve should sit above the Monte Carlo estimate when Jensen's inequality is visible.</p>
{svg_line_chart(scaling, 'd', ['monte_carlo','finite_theory','mean_field'], 'Dimension scaling: Monte Carlo vs formulas', 'expected loss')}
<p><b>How to read it:</b> the blue line is simulated spherical distractors, red is the fixed-vocabulary expansion, and green is the large-vocabulary mean-field closure. Agreement between blue and red supports the theorem; a green line above blue is the Jensen upper-bound effect.</p>
{svg_line_chart(scaling, 'd', ['abs_error','jensen_gap'], 'Approximation diagnostics', 'loss difference')}
<p><b>How to read it:</b> shrinking absolute error indicates the omitted terms decay with dimension. Positive Jensen gap means replacing the random partition sum by its mean overestimates expected loss.</p>
{svg_line_chart(margins, 'margin_m', ['baseline_softplus','p0','variance_coefficient'], 'Margin regimes', 'value')}
<p><b>How to read it:</b> negative margins are hard contexts and have large <code>P0</code>, so the variance coefficient is large; positive margins are easy contexts and suppress distractor variance.</p>
{svg_line_chart(quant, 'sigma_q2', ['predicted_delta_loss'], 'Added logit-noise / quantization prediction', 'delta loss')}
<p><b>How to read it:</b> the model predicts smooth, approximately linear degradation with added independent zero-mean distractor noise. Abrupt cliffs would falsify the local independent-noise assumption.</p>
{svg_line_chart(arch, 'gamma', ['alpha_model','N_star_compute_exponent','D_star_compute_exponent','loss_compute_exponent'], 'Architecture and compute exponents', 'exponent')}
<p><b>How to read it:</b> <code>gamma</code> converts width dimension to parameter count. The model exponent is conditional: <code>alpha=1/gamma</code>. Compute-optimal exponents additionally require an externally supplied data exponent.</p>
<h2>Generated data files</h2><ul><li><code>dimension_scaling.csv</code></li><li><code>margin_regimes.csv</code></li><li><code>quantization_noise.csv</code></li><li><code>architecture_compute.csv</code></li></ul>
"""
    (out / "index.html").write_text(html_doc, encoding="utf-8")
    print(out / "index.html")

if __name__ == "__main__":
    main()
