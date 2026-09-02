#!/usr/bin/env bash
# End-to-end poetry meter/rhyme dataset, training, and checkpoint benchmark demo.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data/poetry_meter_rhyme"
OUT_DIR="${ROOT_DIR}/out/poetry_meter_rhyme_demo"
DEVICE="cuda"
DTYPE="bfloat16"
MAX_ITERS=1000
CKPT_INTERVAL=250
ACCEPT_RESEARCH_ONLY=false
SKIP_DATA=false
SKIP_TRAIN=false
DATA_ONLY=false
CKPT_PATH=""

usage() {
  cat <<'EOF'
Usage: demos/poetry_meter_rhyme_benchmark.sh [options]

Build and tokenize the research-only gold data, train a small demonstration
model, then evaluate every periodic checkpoint to produce a learning curve.

Options:
  --accept-research-only  Confirm local research use of Haider/Chicago annotations.
  --skip-data             Reuse existing generated and tokenized dataset files.
  --skip-train            Do not train; evaluate --ckpt or checkpoints in --out-dir.
  --data-only             Build/tokenize the dataset and stop before training.
  --ckpt PATH             Evaluate one existing checkpoint (implies --skip-train).
  --out-dir PATH          Training results directory (default: out/poetry_meter_rhyme_demo).
  --device DEVICE         Training/evaluation device (default: cuda).
  --dtype DTYPE           float32, float16, or bfloat16 (default: bfloat16).
  --max-iters N           Demonstration training iterations (default: 1000).
  --ckpt-interval N       Save/evaluate interval (default: 250).
  -h, --help              Show this help.

Examples:
  demos/poetry_meter_rhyme_benchmark.sh --accept-research-only
  demos/poetry_meter_rhyme_benchmark.sh --skip-data --device cpu --dtype float32
  demos/poetry_meter_rhyme_benchmark.sh --skip-data --ckpt out/run/ckpt.pt
EOF
}

while (($#)); do
  case "$1" in
    --accept-research-only) ACCEPT_RESEARCH_ONLY=true; shift ;;
    --skip-data) SKIP_DATA=true; shift ;;
    --skip-train) SKIP_TRAIN=true; shift ;;
    --data-only) DATA_ONLY=true; shift ;;
    --ckpt) CKPT_PATH="${2:?--ckpt requires a path}"; SKIP_TRAIN=true; shift 2 ;;
    --out-dir) OUT_DIR="${2:?--out-dir requires a path}"; shift 2 ;;
    --device) DEVICE="${2:?--device requires a value}"; shift 2 ;;
    --dtype) DTYPE="${2:?--dtype requires a value}"; shift 2 ;;
    --max-iters) MAX_ITERS="${2:?--max-iters requires a value}"; shift 2 ;;
    --ckpt-interval) CKPT_INTERVAL="${2:?--ckpt-interval requires a value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${DATA_ONLY}" == true && "${SKIP_DATA}" == true ]]; then
  echo "--data-only and --skip-data cannot be used together." >&2
  exit 2
fi

cd "${ROOT_DIR}"

if [[ "${SKIP_DATA}" == false ]]; then
  if [[ "${ACCEPT_RESEARCH_ONLY}" != true ]]; then
    echo "Refusing research-only downloads without --accept-research-only." >&2
    echo "Review data/poetry_meter_rhyme/README.md before accepting." >&2
    exit 2
  fi
  echo "=== 1/4 Download, build, and validate explicit splits ==="
  (cd "${DATA_DIR}" && ./get_dataset.sh --accept-research-only)

  echo "=== 2/4 Tokenize explicit train/validation files ==="
  (cd "${DATA_DIR}" && python3 prepare.py \
    -t generated/train.txt \
    -v generated/val.txt \
    --method tiktoken \
    --tiktoken_encoding gpt2 \
    --additional_tokens_file special_tokens.json \
    --track_token_counts)
fi

for required in train.bin val.bin meta.pkl generated/test.jsonl; do
  if [[ ! -f "${DATA_DIR}/${required}" ]]; then
    echo "Missing ${DATA_DIR}/${required}; rerun without --skip-data." >&2
    exit 1
  fi
done

if [[ "${DATA_ONLY}" == true ]]; then
  echo "Dataset is ready in ${DATA_DIR}."
  exit 0
fi

mkdir -p "${OUT_DIR}/benchmark"
if [[ "${SKIP_TRAIN}" == false ]]; then
  echo "=== 3/4 Train a compact model and save learning-curve checkpoints ==="
  python3 train.py \
    --dataset poetry_meter_rhyme \
    --out_dir "${OUT_DIR}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --n_layer 4 \
    --n_head 4 \
    --n_embd 256 \
    --block_size 256 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --max_iters "${MAX_ITERS}" \
    --eval_interval "${CKPT_INTERVAL}" \
    --eval_iters 40 \
    --save_major_ckpt_interval "${CKPT_INTERVAL}" \
    --always_save_checkpoint \
    --no-compile
fi

if [[ -n "${CKPT_PATH}" ]]; then
  checkpoints=("${CKPT_PATH}")
else
  mapfile -t checkpoints < <(find "${OUT_DIR}" -maxdepth 1 -type f \
    \( -name 'ckpt.pt' -o -name '[0-9]*.pt' \) -print | sort -V)
fi
if ((${#checkpoints[@]} == 0)); then
  echo "No checkpoints found in ${OUT_DIR}." >&2
  exit 1
fi

echo "=== 4/4 Evaluate all checkpoints on held-out minimal pairs ==="
for checkpoint in "${checkpoints[@]}"; do
  checkpoint_name="$(basename "${checkpoint}" .pt)"
  echo "Evaluating ${checkpoint}"
  python3 benchmarks/evaluate_meter_rhyme.py \
    --ckpt_path "${checkpoint}" \
    --out_dir "${OUT_DIR}" \
    --benchmark_file "${DATA_DIR}/generated/test.jsonl" \
    --tasks meter_minimal_pair,rhyme_minimal_pair \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --length_norm \
    --output_json "${OUT_DIR}/benchmark/${checkpoint_name}.json"
done

echo "Benchmark learning-curve JSON files: ${OUT_DIR}/benchmark/"
