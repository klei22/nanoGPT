#!/bin/bash
# demos/ptq_checkpoint_comparison_demo.sh
#
# End-to-end demonstration of training a Shakespeare character model, applying
# post-training quantization (PTQ) at multiple bit-widths, and comparing each
# quantized checkpoint against the original fp32 weights with the regex
# explorer's L2, angle, and cosine similarity reporting utilities.

set -euo pipefail

DATA_DIR="data/shakespeare_char"
OUT_DIR="out/ptq_shakespeare_comparison"
QUANT_ROOT="${OUT_DIR}_quantized"
ANALYSIS_ROOT="${OUT_DIR}_checkpoint_analysis"
BASELINE_CKPT="${OUT_DIR}/ckpt.pt"
HISTOGRAM_BINS=80

# Quantization bit-widths to evaluate.
declare -a BITS=(8 4)

# Regex patterns and labels to examine with the checkpoint explorer.
declare -a PATTERN_LABELS=(
  "attention_projections"
  "mlp_projections"
  "token_embeddings"
)

declare -a PATTERNS=(
  'transformer\\.h\\.[0-9]+\\.attn\\.(c_attn|c_proj)\\.weight'
  'transformer\\.h\\.[0-9]+\\.mlp\\.(c_fc|c_proj)\\.weight'
  'transformer\\.wte\\.weight'
)

if [ "${#PATTERN_LABELS[@]}" -ne "${#PATTERNS[@]}" ]; then
  echo "Pattern label and pattern arrays must have the same length." >&2
  exit 1
fi

# Ensure matplotlib is available so histogram exports succeed.
if ! python3 -c "import matplotlib" >/dev/null 2>&1; then
  echo "matplotlib is required for histogram export. Install it with 'pip install matplotlib'." >&2
  exit 1
fi

mkdir -p "$DATA_DIR"

step=1
echo "=== Step ${step}: Download the shakespeare_char dataset ==="
pushd "$DATA_DIR" > /dev/null
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_dataset.sh
else
  echo "Found existing tokenized dataset artifacts."
fi
popd > /dev/null

mkdir -p "$OUT_DIR"

step=$((step + 1))
echo "=== Step ${step}: Train a baseline fp32 Shakespeare character model ==="
if [ ! -f "$BASELINE_CKPT" ]; then
  python3 train.py \
    --dataset shakespeare_char \
    --out_dir "$OUT_DIR" \
    --n_layer 4 \
    --n_head 4 \
    --n_embd 256 \
    --block_size 128 \
    --batch_size 64 \
    --max_iters 600 \
    --lr_decay_iters 600 \
    --eval_interval 100 \
    --eval_iters 100 \
    --log_interval 10 \
    --always_save_checkpoint
else
  echo "Found existing baseline checkpoint at $BASELINE_CKPT; skipping training."
fi

if [ ! -f "$BASELINE_CKPT" ]; then
  echo "Expected baseline checkpoint not found at $BASELINE_CKPT" >&2
  exit 1
fi

step=$((step + 1))
echo "=== Step ${step}: Evaluate the baseline checkpoint ==="
python3 sample.py \
  --out_dir "$OUT_DIR" \
  --init_from resume \
  --eval_only \
  --eval_iters 200 \
  --eval_dataset shakespeare_char

mkdir -p "$QUANT_ROOT"

for bit in "${BITS[@]}"; do
  quant_dir="${QUANT_ROOT}/${bit}bit"
  quant_ckpt="${quant_dir}/ckpt.pt"

  step=$((step + 1))
  echo "=== Step ${step}: Quantize to ${bit}-bit weights ==="
  if [ ! -f "$quant_ckpt" ]; then
    python3 quantizations/ptq/fake_quantize_ckpt.py "$OUT_DIR" \
      --out_dir "$quant_dir" \
      --num_bits "$bit"
  else
    echo "Found existing ${bit}-bit checkpoint at $quant_ckpt; skipping quantization."
  fi

  if [ ! -f "$quant_ckpt" ]; then
    echo "Quantized checkpoint not found at $quant_ckpt" >&2
    exit 1
  fi

  step=$((step + 1))
  echo "=== Step ${step}: Evaluate the ${bit}-bit checkpoint ==="
  python3 sample.py \
    --out_dir "$quant_dir" \
    --init_from resume \
    --eval_only \
    --eval_iters 200 \
    --eval_dataset shakespeare_char

done

mkdir -p "$ANALYSIS_ROOT"

step=$((step + 1))
echo "=== Step ${step}: Compare quantized checkpoints against the baseline with the regex explorer ==="
for idx in "${!PATTERNS[@]}"; do
  label="${PATTERN_LABELS[$idx]}"
  pattern="${PATTERNS[$idx]}"
  echo "--- Analyzing pattern '${pattern}' (${label}) ---"

  for bit in "${BITS[@]}"; do
    quant_dir="${QUANT_ROOT}/${bit}bit"
    quant_ckpt="${quant_dir}/ckpt.pt"
    comparison_root="${ANALYSIS_ROOT}/${label}/${bit}bit"
    histogram_dir="${comparison_root}/histograms"
    csv_path="${comparison_root}/comparison_stats.csv"

    echo "Running comparison against ${bit}-bit checkpoint..."
    python3 analysis/checkpoint_analysis/checkpoint_regex_explorer.py \
      "$BASELINE_CKPT" \
      "$pattern" \
      --histogram-dir "$histogram_dir" \
      --histogram-bins "$HISTOGRAM_BINS" \
      --compare-ckpt "$quant_ckpt" \
      --comparison-csv "$csv_path"
  done

done

echo "All comparison tables, histograms, and CSV summaries saved under $ANALYSIS_ROOT."
