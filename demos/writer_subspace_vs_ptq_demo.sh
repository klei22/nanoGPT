#!/usr/bin/env bash
# demos/writer_subspace_vs_ptq_demo.sh
#
# Train (or reuse) a compact Shakespeare-char checkpoint, then compare validation
# loss at several bit widths for:
#   1. the regular full-precision checkpoint,
#   2. existing fake PTQ checkpoints, and
#   3. writer-subspace checkpoints with fake-quantized coefficients.
#
# The summary CSV includes checkpoint size/compression ratio, and the generated
# plot uses bits on the x axis and validation loss on the y axis with one legend
# entry per line.

set -euo pipefail

DATASET="shakespeare_char"
DATASET_DIR="data/${DATASET}"
BASE_OUT_DIR="out_writer_subspace_vs_ptq_${DATASET}"
SWEEP_ROOT="${BASE_OUT_DIR}_sweep"
EVAL_ROOT="${SWEEP_ROOT}/evals"
SUMMARY_ROOT="${SWEEP_ROOT}/summary"
EVAL_ITERS=200
BATCH_SIZE=64
BLOCK_SIZE=256
MAX_ITERS=800

BIT_START=8
BIT_STOP=3
BIT_STEP=-1
WRITER_ATTN_RANK=128
WRITER_MLP_RANK=128
WRITER_VOCAB_RANK=0
PTQ_QUANTIZATIONS="symmetric asymmetric"
PTQ_VECTOR_GROUP_COUNTS="0 2 4"
DEVICE="cuda"
DTYPE="bfloat16"

usage() {
  cat <<'EOF'
Usage: demos/writer_subspace_vs_ptq_demo.sh [options]

Options:
  --bit-start N          Starting bit width for sweeps (default: 8)
  --bit-stop N           Final bit width for sweeps (default: 3)
  --bit-step N           Step increment for sweeps (default: -1)
  --writer-attn-rank N   Rank for block.attn.c_proj writer subspaces (default: 128)
  --writer-mlp-rank N    Rank for block.mlp.c_proj writer subspaces (default: 128)
  --writer-vocab-rank N  Rank for tied vocabulary subspace; 0 disables (default: 0)
  --eval-iters N         Validation batches per evaluation (default: 200)
  --max-iters N          Training iterations for baseline checkpoint (default: 800)
  --ptq-quantizations Q  Space/comma-separated schemes: symmetric/asymmetric (default: "symmetric asymmetric")
  --ptq-group-counts G   Space/comma-separated vector group counts; 0 means per-tensor (default: "0 2 4")
  --device DEVICE        Device passed to train.py/sample.py (default: cuda)
  --dtype DTYPE          Dtype passed to sample.py (default: bfloat16)
  --help                 Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bit-start) BIT_START="$2"; shift 2 ;;
    --bit-stop) BIT_STOP="$2"; shift 2 ;;
    --bit-step) BIT_STEP="$2"; shift 2 ;;
    --writer-attn-rank) WRITER_ATTN_RANK="$2"; shift 2 ;;
    --writer-mlp-rank) WRITER_MLP_RANK="$2"; shift 2 ;;
    --writer-vocab-rank) WRITER_VOCAB_RANK="$2"; shift 2 ;;
    --eval-iters) EVAL_ITERS="$2"; shift 2 ;;
    --max-iters) MAX_ITERS="$2"; shift 2 ;;
    --ptq-quantizations) PTQ_QUANTIZATIONS="${2//,/ }"; shift 2 ;;
    --ptq-group-counts) PTQ_VECTOR_GROUP_COUNTS="${2//,/ }"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if ! mapfile -t BITS < <(seq "$BIT_START" "$BIT_STEP" "$BIT_STOP"); then
  echo "Failed to generate bit-width sweep with start=$BIT_START step=$BIT_STEP stop=$BIT_STOP" >&2
  exit 1
fi
if [[ "${#BITS[@]}" -eq 0 ]]; then
  echo "Bit-width sweep is empty; adjust --bit-start/--bit-stop/--bit-step." >&2
  exit 1
fi

read -r -a PTQ_QUANTIZATION_ARRAY <<< "$PTQ_QUANTIZATIONS"
read -r -a PTQ_GROUP_COUNT_ARRAY <<< "$PTQ_VECTOR_GROUP_COUNTS"
echo "Sweeping bits: ${BITS[*]}"
echo "Writer ranks: attn=${WRITER_ATTN_RANK}, mlp=${WRITER_MLP_RANK}, vocab=${WRITER_VOCAB_RANK}"
echo "Fake PTQ quantization schemes: ${PTQ_QUANTIZATION_ARRAY[*]}"
echo "Fake PTQ vector group counts (0 = per-tensor): ${PTQ_GROUP_COUNT_ARRAY[*]}"

mkdir -p "$DATASET_DIR" "$SWEEP_ROOT" "$EVAL_ROOT" "$SUMMARY_ROOT"

echo "=== Step 1: Prepare ${DATASET} ==="
pushd "$DATASET_DIR" >/dev/null
if [[ ! -f train.bin || ! -f val.bin || ! -f meta.pkl ]]; then
  bash get_dataset.sh
else
  echo "Found existing tokenized dataset artifacts."
fi
popd >/dev/null

echo "=== Step 2: Train/reuse full-precision baseline checkpoint ==="
if [[ ! -f "${BASE_OUT_DIR}/ckpt.pt" ]]; then
  python3 train.py \
    --dataset "$DATASET" \
    --out_dir "$BASE_OUT_DIR" \
    --device "$DEVICE" \
    --n_layer 5 \
    --n_head 3 \
    --n_embd 384 \
    --use_rotary_embeddings \
    --no-use_abs_pos_embeddings \
    --block_size "$BLOCK_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --eval_iters "$EVAL_ITERS" \
    --attention_variant infinite \
    --n_qk_head_dim 120 \
    --n_v_head_dim 120 \
    --use_concat_heads \
    --use_peri_ln \
    --use_qk_norm \
    --use_qk_norm_scale \
    --max_iters "$MAX_ITERS" \
    --eval_interval "$MAX_ITERS" \
    --compile
else
  echo "Found ${BASE_OUT_DIR}/ckpt.pt; skipping training."
fi

run_eval() {
  local ckpt_dir="$1"
  local eval_dir="$2"
  mkdir -p "$eval_dir"
  python3 sample.py \
    --out_dir "$ckpt_dir" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --eval_only \
    --eval_dataset "$DATASET" \
    --eval_iters "$EVAL_ITERS" \
    --eval_output_dir "$eval_dir" \
    --no-print_model_info
}

echo "=== Step 3: Evaluate full-precision baseline ==="
run_eval "$BASE_OUT_DIR" "${EVAL_ROOT}/full_precision/fp32"

echo "=== Step 4: Create/evaluate fake PTQ checkpoints ==="
for scheme in "${PTQ_QUANTIZATION_ARRAY[@]}"; do
  for group_count in "${PTQ_GROUP_COUNT_ARRAY[@]}"; do
    if [[ "$group_count" == "0" ]]; then
      group_tag="tensor"
      extra_ptq_args=(--granularity tensor)
    else
      group_tag="vector_g${group_count}"
      extra_ptq_args=(--granularity vector --vector-group-count "$group_count")
    fi

    for bit in "${BITS[@]}"; do
      ptq_dir="${SWEEP_ROOT}/fake_ptq/${scheme}/${group_tag}/${bit}bit"
      if [[ ! -f "${ptq_dir}/ckpt.pt" ]]; then
        mkdir -p "$ptq_dir"
        python3 quantizations/ptq/fake_quantize_ckpt.py "$BASE_OUT_DIR" \
          --out_dir "$ptq_dir" \
          --num_bits "$bit" \
          --quantization "$scheme" \
          "${extra_ptq_args[@]}"
      else
        echo "Found ${ptq_dir}/ckpt.pt; skipping fake PTQ conversion."
      fi
      run_eval "$ptq_dir" "${EVAL_ROOT}/fake_ptq/${scheme}/${group_tag}/${bit}bit"
    done
  done
done

echo "=== Step 5: Create/evaluate writer-subspace checkpoints ==="
for bit in "${BITS[@]}"; do
  writer_dir="${SWEEP_ROOT}/writer_subspace/${bit}bit"
  if [[ ! -f "${writer_dir}/ckpt.pt" ]]; then
    mkdir -p "$writer_dir"
    python3 - "$BASE_OUT_DIR" "$writer_dir" "$WRITER_ATTN_RANK" "$WRITER_MLP_RANK" "$WRITER_VOCAB_RANK" "$bit" <<'PY'
import os
import shutil
import sys
import torch

from gpt_conf import GPTConfig
from model import GPT
from writer_subspace import SubspaceVocab

base_dir, out_dir = sys.argv[1], sys.argv[2]
attn_rank, mlp_rank, vocab_rank, bits = map(int, sys.argv[3:7])
ckpt_path = os.path.join(base_dir, "ckpt.pt")
checkpoint = torch.load(ckpt_path, map_location="cpu")
model_args = dict(checkpoint["model_args"])

# Build the dense model first, load dense weights, then factor writers. This
# mirrors the intended conversion order and avoids trying to load dense tensors
# into already-factorized module names.
build_args = dict(model_args)
for key in ("writer_attn_rank", "writer_mlp_rank", "writer_vocab_rank"):
    build_args[key] = 0
build_args["writer_coeff_bits"] = 16

model = GPT(GPTConfig(**build_args))
state_dict = checkpoint["model"]
state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

model.config.writer_attn_rank = attn_rank
model.config.writer_mlp_rank = mlp_rank
model.config.writer_vocab_rank = vocab_rank
model.config.writer_coeff_bits = bits
model.convert_writer_subspaces()
if vocab_rank:
    model.transformer.wte = SubspaceVocab.from_embedding(
        model.transformer.wte,
        vocab_rank,
        bits=bits,
    )
    model.lm_head = None

model_args.update(
    writer_attn_rank=attn_rank,
    writer_mlp_rank=mlp_rank,
    writer_vocab_rank=vocab_rank,
    writer_coeff_bits=bits,
)
converted = dict(checkpoint)
converted["model"] = model.state_dict()
converted["model_args"] = model_args
converted["optimizer"] = None
converted["scheduler"] = None
os.makedirs(out_dir, exist_ok=True)
torch.save(converted, os.path.join(out_dir, "ckpt.pt"))
for name in ("meta.pkl", "full_config.json"):
    src = os.path.join(base_dir, name)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(out_dir, name))
PY
  else
    echo "Found ${writer_dir}/ckpt.pt; skipping writer-subspace conversion."
  fi
  run_eval "$writer_dir" "${EVAL_ROOT}/writer_subspace/${bit}bit"
done

python3 - "$BASE_OUT_DIR" "$SWEEP_ROOT" "$EVAL_ROOT" "$SUMMARY_ROOT" "$PTQ_QUANTIZATIONS" "$PTQ_VECTOR_GROUP_COUNTS" "${BITS[@]}" <<'PY'
import csv
import json
import os
import sys
import torch

base_dir = os.path.abspath(sys.argv[1])
sweep_root = os.path.abspath(sys.argv[2])
eval_root = os.path.abspath(sys.argv[3])
summary_root = os.path.abspath(sys.argv[4])
ptq_quantizations = sys.argv[5].split()
ptq_group_counts = [int(value) for value in sys.argv[6].split()]
bits = [int(arg) for arg in sys.argv[7:]]

os.makedirs(summary_root, exist_ok=True)
base_ckpt = os.path.join(base_dir, "ckpt.pt")
base_size = os.path.getsize(base_ckpt)

def estimated_quantized_mb(ckpt_path, bits, method):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    total = 0.0
    for name, tensor in state.items():
        if not torch.is_tensor(tensor):
            continue
        original = tensor.numel() * tensor.element_size()
        if not torch.is_floating_point(tensor):
            total += original
        elif method == "fake_ptq":
            total += tensor.numel() * bits / 8.0
        elif method == "writer_subspace" and (name.endswith("down.weight") or ".coeff.weight" in name):
            total += tensor.numel() * bits / 8.0
        else:
            total += original
    return total / (1024 * 1024)

base_estimated_mb = estimated_quantized_mb(base_ckpt, 32, "fake_ptq")

def read_loss(path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if "val" not in data:
        raise SystemExit(f"No 'val' key found in {path}")
    return float(data["val"])

rows = []
rows.append({
    "bits": 32,
    "method": "Full precision",
    "quantization": "fp32",
    "granularity": "dense",
    "group_count": "",
    "label": "Full precision",
    "val_loss": read_loss(os.path.join(eval_root, "full_precision", "fp32", "eval_loss.txt")),
    "ckpt_mb": base_size / (1024 * 1024),
    "estimated_quantized_mb": base_estimated_mb,
    "compression_ratio_vs_fp32": 1.0,
})

for scheme in ptq_quantizations:
    for group_count in ptq_group_counts:
        group_tag = "tensor" if group_count == 0 else f"vector_g{group_count}"
        granularity = "tensor" if group_count == 0 else "vector"
        label = f"Fake PTQ {scheme} {granularity}"
        if group_count:
            label += f" g={group_count}"
        for bit in bits:
            ckpt_path = os.path.join(sweep_root, "fake_ptq", scheme, group_tag, f"{bit}bit", "ckpt.pt")
            eval_path = os.path.join(eval_root, "fake_ptq", scheme, group_tag, f"{bit}bit", "eval_loss.txt")
            size = os.path.getsize(ckpt_path)
            rows.append({
                "bits": bit,
                "method": "Fake PTQ",
                "quantization": scheme,
                "granularity": granularity,
                "group_count": group_count,
                "label": label,
                "val_loss": read_loss(eval_path),
                "ckpt_mb": size / (1024 * 1024),
                "estimated_quantized_mb": estimated_quantized_mb(ckpt_path, bit, "fake_ptq"),
                "compression_ratio_vs_fp32": base_estimated_mb / estimated_quantized_mb(ckpt_path, bit, "fake_ptq"),
            })

for bit in bits:
    ckpt_path = os.path.join(sweep_root, "writer_subspace", f"{bit}bit", "ckpt.pt")
    eval_path = os.path.join(eval_root, "writer_subspace", f"{bit}bit", "eval_loss.txt")
    size = os.path.getsize(ckpt_path)
    rows.append({
        "bits": bit,
        "method": "Writer subspace",
        "quantization": "symmetric",
        "granularity": "writer_coeff",
        "group_count": "",
        "label": "Writer subspace symmetric coeffs",
        "val_loss": read_loss(eval_path),
        "ckpt_mb": size / (1024 * 1024),
        "estimated_quantized_mb": estimated_quantized_mb(ckpt_path, bit, "writer_subspace"),
        "compression_ratio_vs_fp32": base_estimated_mb / estimated_quantized_mb(ckpt_path, bit, "writer_subspace"),
    })

csv_path = os.path.join(summary_root, "writer_subspace_vs_ptq_eval.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(
        fh,
        fieldnames=["bits", "method", "quantization", "granularity", "group_count", "label", "val_loss", "ckpt_mb", "estimated_quantized_mb", "compression_ratio_vs_fp32"],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({
            "bits": row["bits"],
            "method": row["method"],
            "quantization": row["quantization"],
            "granularity": row["granularity"],
            "group_count": row["group_count"],
            "label": row["label"],
            "val_loss": f"{row['val_loss']:.8f}",
            "ckpt_mb": f"{row['ckpt_mb']:.4f}",
            "estimated_quantized_mb": f"{row['estimated_quantized_mb']:.4f}",
            "compression_ratio_vs_fp32": f"{row['compression_ratio_vs_fp32']:.4f}",
        })
print(f"Wrote summary CSV to {csv_path}")

def color_for(index):
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    return palette[index % len(palette)]

def make_svg(points_by_label, x_key, y_key, x_label, y_label, title, full_precision_loss=None):
    width, height = 1200, 460
    left, right, top, bottom = 80, 320, 45, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    all_points = [point for points in points_by_label.values() for point in points]
    xs = [float(point[x_key]) for point in all_points]
    ys = [float(point[y_key]) for point in all_points]
    if full_precision_loss is not None:
        ys.append(float(full_precision_loss))
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmin == xmax:
        xmin -= 0.5; xmax += 0.5
    if ymin == ymax:
        ymin -= 0.5; ymax += 0.5
    ypad = (ymax - ymin) * 0.08
    ymin -= ypad; ymax += ypad

    def sx(x):
        # Keep quantization plots in the conventional high-to-low bit direction.
        if x_key == "bits":
            return left + (xmax - float(x)) / (xmax - xmin) * plot_w
        return left + (float(x) - xmin) / (xmax - xmin) * plot_w
    def sy(y):
        return top + (ymax - float(y)) / (ymax - ymin) * plot_h

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{title}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="{width/2}" y="24" text-anchor="middle" font-size="18" font-family="sans-serif">{title}</text>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333"/>')
    for i in range(6):
        x = xmin + (xmax - xmin) * i / 5
        px = sx(x)
        label = f"{x:.0f}" if x_key == "bits" else f"{x:.2f}"
        parts.append(f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{top+plot_h}" stroke="#ddd"/>')
        parts.append(f'<text x="{px:.1f}" y="{top+plot_h+22}" text-anchor="middle" font-size="11" font-family="sans-serif">{label}</text>')
    for i in range(6):
        y = ymin + (ymax - ymin) * i / 5
        py = sy(y)
        parts.append(f'<line x1="{left}" y1="{py:.1f}" x2="{left+plot_w}" y2="{py:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{left-8}" y="{py+4:.1f}" text-anchor="end" font-size="11" font-family="sans-serif">{y:.3f}</text>')
    if full_precision_loss is not None:
        y = sy(full_precision_loss)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#111" stroke-dasharray="4 4"/>')
        parts.append(f'<text x="{left+plot_w-4}" y="{y-6:.1f}" text-anchor="end" font-size="11" font-family="sans-serif">Full precision</text>')
    legend_y = top + 10
    for idx, (label, points) in enumerate(points_by_label.items()):
        color = color_for(idx)
        points = sorted(points, key=lambda point: point[x_key], reverse=(x_key == "bits"))
        coords = " ".join(f'{sx(point[x_key]):.1f},{sy(point[y_key]):.1f}' for point in points)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        for point in points:
            parts.append(f'<circle cx="{sx(point[x_key]):.1f}" cy="{sy(point[y_key]):.1f}" r="3" fill="{color}"><title>{label}: bits={point["bits"]}, loss={point["val_loss"]:.6f}, estimated size={point["estimated_quantized_mb"]:.3f} MB</title></circle>')
        lx = left + plot_w + 10
        ly = legend_y + idx * 18
        # If the legend would overflow, wrap it into the lower left of the SVG.
        if ly > height - 25:
            lx = left + 10
            ly = top + plot_h - 18 * (len(points_by_label) - idx)
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+18}" y2="{ly}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{lx+23}" y="{ly+4}" font-size="11" font-family="sans-serif">{label}</text>')
    parts.append(f'<text x="{left+plot_w/2}" y="{height-24}" text-anchor="middle" font-size="13" font-family="sans-serif">{x_label}</text>')
    parts.append(f'<text x="18" y="{top+plot_h/2}" transform="rotate(-90 18 {top+plot_h/2})" text-anchor="middle" font-size="13" font-family="sans-serif">{y_label}</text>')
    parts.append('</svg>')
    return "\n".join(parts)

full_precision = next(row for row in rows if row["method"] == "Full precision")
quantized_rows = [row for row in rows if row["method"] != "Full precision"]
points_by_label = {}
for row in quantized_rows:
    points_by_label.setdefault(row["label"], []).append(row)

bits_svg = make_svg(
    points_by_label,
    "bits",
    "val_loss",
    "Bits",
    "Validation loss",
    "Validation loss vs quantization bits",
    full_precision_loss=full_precision["val_loss"],
)
size_svg = make_svg(
    points_by_label,
    "estimated_quantized_mb",
    "val_loss",
    "Estimated quantized size (MB)",
    "Validation loss",
    "Validation loss vs estimated quantized size",
    full_precision_loss=full_precision["val_loss"],
)

html_path = os.path.join(summary_root, "writer_subspace_vs_ptq_eval.html")
rows_html = "\n".join(
    "<tr>" + "".join(
        f"<td>{row[column]}</td>" if column not in {"val_loss", "ckpt_mb", "estimated_quantized_mb", "compression_ratio_vs_fp32"}
        else f"<td>{float(row[column]):.6f}</td>"
        for column in ["bits", "method", "quantization", "granularity", "group_count", "label", "val_loss", "ckpt_mb", "estimated_quantized_mb", "compression_ratio_vs_fp32"]
    ) + "</tr>"
    for row in rows
)
with open(html_path, "w", encoding="utf-8") as fh:
    fh.write(f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Writer-subspace vs fake PTQ evaluation</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; }}
    .chart {{ border: 1px solid #ddd; margin: 1.5rem 0; overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
    th, td {{ border: 1px solid #ddd; padding: 0.35rem 0.5rem; text-align: left; }}
    th {{ background: #f4f4f4; }}
  </style>
</head>
<body>
  <h1>Writer-subspace vs fake PTQ evaluation</h1>
  <p>Full precision is shown as a dotted horizontal reference line so that the bit-width axis stays focused on quantized settings.</p>
  <div class=\"chart\">{bits_svg}</div>
  <div class=\"chart\">{size_svg}</div>
  <h2>All stats</h2>
  <table>
    <thead><tr><th>Bits</th><th>Method</th><th>Quantization</th><th>Granularity</th><th>Group count</th><th>Label</th><th>Validation loss</th><th>Checkpoint MB</th><th>Estimated quantized MB</th><th>Compression ratio vs FP32</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</body>
</html>
""")
print(f"Wrote HTML report to {html_path}")

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is not installed; skipping PNG plot generation.")
else:
    for x_key, x_label, title, filename in (
        ("bits", "Bits", "Validation loss vs quantization bits", "writer_subspace_vs_ptq_eval_bits.png"),
        ("estimated_quantized_mb", "Estimated quantized size (MB)", "Validation loss vs estimated quantized size", "writer_subspace_vs_ptq_eval_size.png"),
    ):
        plt.figure(figsize=(9.5, 5.5))
        for label, points in points_by_label.items():
            points = sorted(points, key=lambda row: row[x_key], reverse=(x_key == "bits"))
            plt.plot([row[x_key] for row in points], [row["val_loss"] for row in points], marker="o", label=label)
        plt.axhline(full_precision["val_loss"], linestyle=":", color="black", label="Full precision")
        if x_key == "bits":
            plt.xticks(sorted({row["bits"] for row in quantized_rows}, reverse=True))
            plt.gca().invert_xaxis()
        plt.xlabel(x_label)
        plt.ylabel("Validation loss")
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend(fontsize="small")
        plt.tight_layout()
        png_path = os.path.join(summary_root, filename)
        plt.savefig(png_path, dpi=200)
        print(f"Wrote plot to {png_path}")
        plt.close()
PY

echo "Demo complete. Summary artifacts are in ${SUMMARY_ROOT}."
