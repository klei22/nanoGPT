"""
ANN vs full-vocab LM-head “crossing point” estimator
----------------------------------------------------
What it estimates (per generated token):
  - Latency (LM-head only AND optional total decode)
  - FLOPs (approx)
  - Memory traffic (bytes read for weights, approx)
  - Memory footprint (LM-head weights, ANN index, KV cache)
  - Power & energy per token (E = avg_power * time)

Key idea:
  Full vocab logits:  logits = W @ h     with W shape (V, d)
  ANN approach:       retrieve candidates (~M), rerank exactly on those M

This is a *back-of-the-envelope* model:
  - Effective bandwidth & TFLOPs should be treated as “measured-in-practice” numbers.
  - ANN recall needs can force M much larger than k; that’s why alpha matters.
  - Power is a user-provided average power during decode.

Dependencies:
  pip install numpy plotly
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ----------------------------
# Hardware presets (EDIT ME)
# ----------------------------
@dataclass(frozen=True)
class Hardware:
    name: str
    BW_seq_GBs: float       # effective sequential bandwidth (dense streaming)
    BW_rand_GBs: float      # effective random-ish bandwidth (ANN candidate fetch)
    compute_TFLOPS: float   # effective compute throughput (often not the limiter for GEMV)
    power_W: float          # avg power during decode (for energy estimate)
    t_index_us: float       # ANN index traversal overhead per token
    framework_us: float     # misc overhead (tokenization/sampling/glue code/kernel launch, etc.)


PRESETS: Dict[str, Hardware] = {
    # Edge-ish CPU (laptop / phone class). Tune BW and power to your device.
    "cpu_edge": Hardware(
        name="CPU (edge-ish)",
        BW_seq_GBs=25.0,
        BW_rand_GBs=2.5,      # ~10x worse than streaming is a common placeholder
        compute_TFLOPS=0.25,  # effective for matvec is usually low; mem often dominates
        power_W=8.0,          # average during decode (NOT TDP)
        t_index_us=40.0,
        framework_us=30.0,
    ),
    # Desktop/server CPU class (DDR5). Still usually bandwidth-limited for large V GEMV.
    "cpu_server": Hardware(
        name="CPU (server-ish)",
        BW_seq_GBs=80.0,
        BW_rand_GBs=8.0,
        compute_TFLOPS=2.0,
        power_W=120.0,
        t_index_us=30.0,
        framework_us=15.0,
    ),
    # Consumer discrete GPU (batch=1 decode often has noticeable overheads).
    "gpu_consumer": Hardware(
        name="GPU (consumer-ish)",
        BW_seq_GBs=600.0,
        BW_rand_GBs=60.0,
        compute_TFLOPS=50.0,
        power_W=250.0,
        t_index_us=10.0,
        framework_us=15.0,  # kernel-launch / dispatch overhead can matter at batch=1
    ),
    # Datacenter GPU-ish placeholder.
    "gpu_dc": Hardware(
        name="GPU (datacenter-ish)",
        BW_seq_GBs=1200.0,
        BW_rand_GBs=120.0,
        compute_TFLOPS=200.0,
        power_W=350.0,
        t_index_us=10.0,
        framework_us=10.0,
    ),
}

# Choose what to plot:
PRESETS_TO_PLOT: List[str] = ["cpu_edge", "gpu_consumer"]  # e.g. ["cpu_edge"] or ["gpu_consumer"]


# ----------------------------
# Model + ANN knobs (EDIT ME)
# ----------------------------
# "Gemma-like" defaults from your description:
V_target = 262_144   # vocab size to mark with a vertical line (Gemma-like)
d = 640              # embedding/hidden size

# Quantization / dtype for *weights you stream* in this toy model
# (KV cache is often fp16 even when weights are int8/int4.)
dtype = "int8"       # "fp16", "int8", "int4"
BYTES_PER_WEIGHT = {"fp16": 2.0, "int8": 1.0, "int4": 0.5}
b_w = BYTES_PER_WEIGHT[dtype]
b_kv = 2.0           # KV cache element bytes (often fp16/bf16)

# ANN candidate parameters:
k = 65
alpha = 60           # candidates multiplier: M = alpha * k  (tune this for recall!)
M = int(alpha * k)

# Rough memory overhead of ANN index (HNSW/graph/etc.) per vector.
# This can vary wildly. 64–256 bytes/vector is a reasonable “placeholder range.”
index_bytes_per_vector = 128

# Optional: include “rest of model” (backbone) cost to estimate *total decode*.
# Crossing point for LM-head does NOT change if backbone is identical for both methods,
# but energy/latency per token in decode DOES, so we include it for your request.
P_total = 270_000_000
embed_fraction = 0.628
P_backbone = int(round(P_total * (1.0 - embed_fraction)))  # ~100M if embed_fraction=0.628

# KV cache footprint estimate
num_layers = 18
context_len = 2048  # decode context length you care about (changes KV cache footprint)

# Plot range for vocab size
V_min, V_max = 1_000, 1_000_000
num_points = 500


# ----------------------------
# Core cost model helpers
# ----------------------------
def time_compute_mem(flops: float, bytes_read: float, compute_TFLOPS: float, BW_GBs: float) -> Tuple[float, float, float]:
    """Return (t, t_compute, t_mem) in seconds, using max(compute, memory) model."""
    t_compute = 0.0 if compute_TFLOPS <= 0 else flops / (compute_TFLOPS * 1e12)
    t_mem = 0.0 if BW_GBs <= 0 else bytes_read / (BW_GBs * 1e9)
    return max(t_compute, t_mem), t_compute, t_mem


def lm_head_dense_time(hw: Hardware, V: np.ndarray) -> np.ndarray:
    """LM-head dense logits time (seconds) over vocab sizes V."""
    flops = 2.0 * V * d
    bytes_read = V * d * b_w
    t = np.zeros_like(V, dtype=float)
    for i in range(len(V)):
        t[i], _, _ = time_compute_mem(flops[i], bytes_read[i], hw.compute_TFLOPS, hw.BW_seq_GBs)
    return t


def lm_head_ann_time(hw: Hardware) -> float:
    """LM-head ANN+rerank time (seconds), constant w.r.t vocab size in this simple model."""
    # Exact scoring of M candidates:
    flops = 2.0 * M * d
    bytes_read = M * d * b_w
    t_score, _, _ = time_compute_mem(flops, bytes_read, hw.compute_TFLOPS, hw.BW_rand_GBs)
    return (hw.t_index_us * 1e-6) + t_score


def backbone_time(hw: Hardware) -> float:
    """
    Rough backbone per-token time (seconds).
    Very rough approximation: flops ~ 2 * P_backbone, bytes_read ~ P_backbone * b_w.
    """
    flops = 2.0 * P_backbone
    bytes_read = P_backbone * b_w
    t, _, _ = time_compute_mem(flops, bytes_read, hw.compute_TFLOPS, hw.BW_seq_GBs)
    return t


def crossing_point_Vstar(hw: Hardware) -> float:
    """
    Solve for V* where LM-head dense time equals LM-head ANN time.

    Dense LM-head time under max(compute, memory) still scales linearly in V:
      t_dense(V) = V * slope, where slope = max(2d/compute, d*b/BW_seq)

    So V* = t_ann / slope.
    """
    # slopes (seconds per vocab item)
    slope_compute = 0.0 if hw.compute_TFLOPS <= 0 else (2.0 * d) / (hw.compute_TFLOPS * 1e12)
    slope_mem = 0.0 if hw.BW_seq_GBs <= 0 else (d * b_w) / (hw.BW_seq_GBs * 1e9)
    slope = max(slope_compute, slope_mem)

    t_ann = lm_head_ann_time(hw)
    if slope <= 0:
        return float("nan")
    return t_ann / slope


# ----------------------------
# Memory footprint helpers
# ----------------------------
def bytes_to_mb(x: float) -> float:
    return x / 1e6

def bytes_to_mib(x: float) -> float:
    return x / (1024**2)

def memory_footprint(V: int) -> Dict[str, float]:
    lm_head_weights = V * d * b_w
    ann_index = V * index_bytes_per_vector
    backbone_weights = P_backbone * b_w
    kv_cache = 2.0 * num_layers * context_len * d * b_kv  # K and V per layer per token
    total_model_weights = lm_head_weights + backbone_weights
    return {
        "lm_head_weights_bytes": lm_head_weights,
        "backbone_weights_bytes": backbone_weights,
        "total_weights_bytes": total_model_weights,
        "ann_index_bytes": ann_index,
        "kv_cache_bytes": kv_cache,
    }


# ----------------------------
# Reporting + Plotting
# ----------------------------
def summarize(hw: Hardware, V_eval: int = V_target) -> None:
    Vstar = crossing_point_Vstar(hw)

    # LM-head times at V_eval
    t_dense_lm = float(lm_head_dense_time(hw, np.array([V_eval]))[0])
    t_ann_lm = lm_head_ann_time(hw)

    # Backbone + framework
    t_back = backbone_time(hw)
    t_misc = hw.framework_us * 1e-6

    # Total per-token decode time (very rough)
    t_dense_total = t_misc + t_back + t_dense_lm
    t_ann_total = t_misc + t_back + t_ann_lm

    # Energy per token
    E_dense = hw.power_W * t_dense_total
    E_ann = hw.power_W * t_ann_total

    # FLOPs + bytes (LM-head)
    flops_dense_lm = 2.0 * V_eval * d
    bytes_dense_lm = V_eval * d * b_w
    flops_ann_lm = 2.0 * M * d
    bytes_ann_lm = M * d * b_w

    mem = memory_footprint(V_eval)

    print("\n" + "=" * 80)
    print(f"{hw.name}  (preset key: {', '.join([k for k,v in PRESETS.items() if v==hw])})")
    print("-" * 80)
    print(f"Effective BW_seq={hw.BW_seq_GBs:.1f} GB/s, BW_rand={hw.BW_rand_GBs:.1f} GB/s, compute={hw.compute_TFLOPS:.2f} TFLOPS")
    print(f"Index overhead t_index={hw.t_index_us:.1f} us, framework/misc={hw.framework_us:.1f} us")
    print(f"Avg power={hw.power_W:.1f} W  (energy = power * time)")

    print("\nModel knobs:")
    print(f"  V_eval={V_eval:,}, d={d}, dtype={dtype} (b_w={b_w} bytes/weight), KV dtype bytes={b_kv}")
    print(f"  k={k}, alpha={alpha} => M={M}")
    print(f"  P_backbone≈{P_backbone:,} params (rough), layers={num_layers}, context_len={context_len}")

    print("\nLM-head (logits) per token @ V_eval:")
    print(f"  Dense:  time={t_dense_lm*1e3:.3f} ms | FLOPs={flops_dense_lm/1e9:.3f} GF | bytes_read≈{bytes_dense_lm/1e6:.1f} MB")
    print(f"  ANN:    time={t_ann_lm*1e3:.3f} ms | FLOPs={flops_ann_lm/1e9:.3f} GF | bytes_read≈{bytes_ann_lm/1e6:.3f} MB  (+ index traversal)")

    print("\nTotal decode (VERY rough: backbone + framework + LM-head) per token:")
    print(f"  Dense total: {t_dense_total*1e3:.3f} ms  => energy/token ≈ {E_dense*1e3:.3f} mJ")
    print(f"  ANN total:   {t_ann_total*1e3:.3f} ms  => energy/token ≈ {E_ann*1e3:.3f} mJ")

    print("\nCrossing point (LM-head only):")
    print(f"  V* ≈ {Vstar:,.0f}  (dense logits faster when V < V*, ANN faster when V > V*)")

    print("\nMemory footprint @ V_eval (static-ish):")
    print(f"  LM-head weights:   {bytes_to_mib(mem['lm_head_weights_bytes']):.1f} MiB")
    print(f"  Backbone weights:  {bytes_to_mib(mem['backbone_weights_bytes']):.1f} MiB")
    print(f"  Total weights:     {bytes_to_mib(mem['total_weights_bytes']):.1f} MiB")
    print(f"  ANN index (guess): {bytes_to_mib(mem['ann_index_bytes']):.1f} MiB  (index_bytes_per_vector={index_bytes_per_vector})")
    print(f"  KV cache:          {bytes_to_mib(mem['kv_cache_bytes']):.1f} MiB  (scales with context_len)")


def make_plots_for_hw(hw: Hardware) -> Tuple[go.Figure, go.Figure]:
    V = np.logspace(np.log10(V_min), np.log10(V_max), num_points).astype(int)
    V = np.unique(V)  # ensure monotonic unique ints

    # LM-head time curves
    t_dense_lm = lm_head_dense_time(hw, V)
    t_ann_lm = lm_head_ann_time(hw) * np.ones_like(V, dtype=float)

    # Total decode time curves (adds constant backbone + misc)
    t_back = backbone_time(hw)
    t_misc = hw.framework_us * 1e-6
    t_dense_total = t_dense_lm + t_back + t_misc
    t_ann_total = t_ann_lm + t_back + t_misc

    # Energy curves (J/token)
    E_dense = hw.power_W * t_dense_total
    E_ann = hw.power_W * t_ann_total

    Vstar = crossing_point_Vstar(hw)

    # ----------------------------
    # TIME FIGURE (2 rows)
    # ----------------------------
    fig_time = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "LM-head (vocab projection) latency vs vocabulary size — ms/token",
            "Total decode latency vs vocabulary size (backbone + framework + LM-head) — ms/token"
        ),
        vertical_spacing=0.12,
    )

    # Trace labels (explicit + consistent)
    fig_time.add_trace(
        go.Scatter(
            x=V, y=t_dense_lm * 1e3, mode="lines",
            name="Dense full-vocab (LM-head)",
            hovertemplate="V=%{x:,}<br>LM-head latency=%{y:.3f} ms/token<extra></extra>",
            legendgroup="lm",
        ),
        row=1, col=1
    )
    fig_time.add_trace(
        go.Scatter(
            x=V, y=t_ann_lm * 1e3, mode="lines",
            name="ANN + rerank (LM-head)",
            hovertemplate="V=%{x:,}<br>LM-head latency=%{y:.3f} ms/token<extra></extra>",
            legendgroup="lm",
        ),
        row=1, col=1
    )

    fig_time.add_trace(
        go.Scatter(
            x=V, y=t_dense_total * 1e3, mode="lines",
            name="Dense full-vocab (TOTAL decode)",
            hovertemplate="V=%{x:,}<br>Total latency=%{y:.3f} ms/token<extra></extra>",
            legendgroup="total",
        ),
        row=2, col=1
    )
    fig_time.add_trace(
        go.Scatter(
            x=V, y=t_ann_total * 1e3, mode="lines",
            name="ANN + rerank (TOTAL decode)",
            hovertemplate="V=%{x:,}<br>Total latency=%{y:.3f} ms/token<extra></extra>",
            legendgroup="total",
        ),
        row=2, col=1
    )

    # Vertical markers (apply across both rows)
    shapes = []
    if np.isfinite(Vstar) and V_min <= Vstar <= V_max:
        shapes.append(dict(
            type="line", x0=Vstar, x1=Vstar, y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(dash="dash")
        ))
    if V_min <= V_target <= V_max:
        shapes.append(dict(
            type="line", x0=V_target, x1=V_target, y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(dash="dot")
        ))
    fig_time.update_layout(shapes=shapes)

    # Clear annotations for the vertical lines
    annotations = []
    if np.isfinite(Vstar) and V_min <= Vstar <= V_max:
        annotations.append(dict(
            x=Vstar, y=1.05, xref="x", yref="paper",
            text=f"Crossing point: V* ≈ {Vstar:,.0f}",
            showarrow=False
        ))
    if V_min <= V_target <= V_max:
        annotations.append(dict(
            x=V_target, y=-0.10, xref="x", yref="paper",
            text=f"Reference vocab: V_target = {V_target:,}",
            showarrow=False
        ))

    # Add a “settings box” so each plot is self-identifying
    settings_text = (
        f"<b>Preset:</b> {hw.name}<br>"
        f"<b>d</b>={d}, <b>dtype</b>={dtype} (b_w={b_w} B/weight)<br>"
        f"<b>k</b>={k}, <b>alpha</b>={alpha} ⇒ <b>M</b>={M}<br>"
        f"<b>BW_seq</b>={hw.BW_seq_GBs:.1f} GB/s, <b>BW_rand</b>={hw.BW_rand_GBs:.1f} GB/s<br>"
        f"<b>t_index</b>={hw.t_index_us:.1f} µs, <b>framework</b>={hw.framework_us:.1f} µs<br>"
        f"<b>power</b>≈{hw.power_W:.1f} W"
    )
    annotations.append(dict(
        x=0.01, y=0.99, xref="paper", yref="paper",
        xanchor="left", yanchor="top",
        text=settings_text,
        showarrow=False,
        borderwidth=1,
        bgcolor="rgba(255,255,255,0.85)",
    ))

    fig_time.update_layout(annotations=annotations)

    fig_time.update_xaxes(type="log", title_text="Vocabulary size V (tokens) — log scale", row=2, col=1)
    fig_time.update_yaxes(title_text="LM-head latency (ms/token)", row=1, col=1)
    fig_time.update_yaxes(title_text="Total latency (ms/token)", row=2, col=1)

    fig_time.update_layout(
        height=820,
        hovermode="x unified",
        legend_title_text="Method (what each line represents)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=dict(
            text="Latency crossover estimate: Dense full-vocab vs ANN+rerank",
            x=0.5
        ),
    )

    # ----------------------------
    # ENERGY FIGURE
    # ----------------------------
    fig_energy = go.Figure()
    fig_energy.add_trace(go.Scatter(
        x=V, y=E_dense * 1e3, mode="lines",
        name="Dense full-vocab (TOTAL energy)",
        hovertemplate="V=%{x:,}<br>Energy=%{y:.3f} mJ/token<extra></extra>",
    ))
    fig_energy.add_trace(go.Scatter(
        x=V, y=E_ann * 1e3, mode="lines",
        name="ANN + rerank (TOTAL energy)",
        hovertemplate="V=%{x:,}<br>Energy=%{y:.3f} mJ/token<extra></extra>",
    ))

    if np.isfinite(Vstar) and V_min <= Vstar <= V_max:
        fig_energy.add_vline(
            x=Vstar, line_dash="dash",
            annotation_text=f"V* ≈ {Vstar:,.0f}",
            annotation_position="top left"
        )
    fig_energy.add_vline(
        x=V_target, line_dash="dot",
        annotation_text=f"V_target = {V_target:,}",
        annotation_position="bottom right"
    )

    fig_energy.update_layout(
        title=dict(
            text=f"Energy per token vs vocabulary size (rough) — {hw.name}",
            x=0.5
        ),
        xaxis_title="Vocabulary size V (tokens) — log scale",
        yaxis_title="Energy per token (mJ/token)",
        xaxis_type="log",
        hovermode="x unified",
        legend_title_text="Method (what each line represents)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=520,
    )

    return fig_time, fig_energy


# ----------------------------
# Run: print + plot for CPU/GPU choices
# ----------------------------
if __name__ == "__main__":
    for key in PRESETS_TO_PLOT:
        hw = PRESETS[key]
        summarize(hw, V_eval=V_target)
        fig_t, fig_e = make_plots_for_hw(hw)
        fig_t.show()
        fig_e.show()

