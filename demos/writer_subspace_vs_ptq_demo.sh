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

echo "Sweeping bits: ${BITS[*]}"
echo "Writer ranks: attn=${WRITER_ATTN_RANK}, mlp=${WRITER_MLP_RANK}, vocab=${WRITER_VOCAB_RANK}"

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
for bit in "${BITS[@]}"; do
  ptq_dir="${SWEEP_ROOT}/fake_ptq/${bit}bit"
  if [[ ! -f "${ptq_dir}/ckpt.pt" ]]; then
    mkdir -p "$ptq_dir"
    python3 quantizations/ptq/fake_quantize_ckpt.py "$BASE_OUT_DIR" \
      --out_dir "$ptq_dir" \
      --num_bits "$bit"
  else
    echo "Found ${ptq_dir}/ckpt.pt; skipping fake PTQ conversion."
  fi
  run_eval "$ptq_dir" "${EVAL_ROOT}/fake_ptq/${bit}bit"
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

python3 - "$BASE_OUT_DIR" "$SWEEP_ROOT" "$EVAL_ROOT" "$SUMMARY_ROOT" "${BITS[@]}" <<'PY'
import csv
import json
import os
import sys

base_dir = os.path.abspath(sys.argv[1])
sweep_root = os.path.abspath(sys.argv[2])
eval_root = os.path.abspath(sys.argv[3])
summary_root = os.path.abspath(sys.argv[4])
bits = [int(arg) for arg in sys.argv[5:]]

os.makedirs(summary_root, exist_ok=True)
base_ckpt = os.path.join(base_dir, "ckpt.pt")
base_size = os.path.getsize(base_ckpt)

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
    "val_loss": read_loss(os.path.join(eval_root, "full_precision", "fp32", "eval_loss.txt")),
    "ckpt_mb": base_size / (1024 * 1024),
    "compression_ratio_vs_fp32": 1.0,
})
for method, ckpt_subdir, eval_subdir in (
    ("Fake PTQ", "fake_ptq", "fake_ptq"),
    ("Writer subspace", "writer_subspace", "writer_subspace"),
):
    for bit in bits:
        ckpt_path = os.path.join(sweep_root, ckpt_subdir, f"{bit}bit", "ckpt.pt")
        eval_path = os.path.join(eval_root, eval_subdir, f"{bit}bit", "eval_loss.txt")
        size = os.path.getsize(ckpt_path)
        rows.append({
            "bits": bit,
            "method": method,
            "val_loss": read_loss(eval_path),
            "ckpt_mb": size / (1024 * 1024),
            "compression_ratio_vs_fp32": base_size / size if size else float("nan"),
        })

csv_path = os.path.join(summary_root, "writer_subspace_vs_ptq_eval.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(
        fh,
        fieldnames=["bits", "method", "val_loss", "ckpt_mb", "compression_ratio_vs_fp32"],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({
            "bits": row["bits"],
            "method": row["method"],
            "val_loss": f"{row['val_loss']:.8f}",
            "ckpt_mb": f"{row['ckpt_mb']:.4f}",
            "compression_ratio_vs_fp32": f"{row['compression_ratio_vs_fp32']:.4f}",
        })
print(f"Wrote summary CSV to {csv_path}")

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is not installed; skipping PNG plot generation.")
else:
    plt.figure(figsize=(8.5, 5.0))
    for method in ("Full precision", "Fake PTQ", "Writer subspace"):
        subset = [row for row in rows if row["method"] == method]
        subset.sort(key=lambda row: row["bits"], reverse=True)
        linestyle = "--" if method == "Full precision" else "-"
        marker = "x" if method == "Full precision" else "o"
        plt.plot(
            [row["bits"] for row in subset],
            [row["val_loss"] for row in subset],
            marker=marker,
            linestyle=linestyle,
            label=method,
        )
    all_bits = sorted({row["bits"] for row in rows}, reverse=True)
    plt.xticks(all_bits)
    plt.gca().invert_xaxis()
    plt.xlabel("Bits")
    plt.ylabel("Validation loss")
    plt.title("Writer-subspace coefficients vs fake PTQ: validation loss by bits")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(title="Method")
    plt.tight_layout()
    png_path = os.path.join(summary_root, "writer_subspace_vs_ptq_eval.png")
    plt.savefig(png_path, dpi=200)
    print(f"Wrote plot to {png_path}")
PY

echo "Demo complete. Summary artifacts are in ${SUMMARY_ROOT}."
