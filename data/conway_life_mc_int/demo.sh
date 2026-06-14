#!/usr/bin/env bash
# End-to-end Conway Life dataset demo:
#   1. generate the CSV and multicontext manifest
#   2. train a tiny multicontext model
#   3. seed sampling with the first rows of the validation split
#   4. append sampled rows to the validation split
#   5. open the viewer preloaded with validation+sample CSV

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PORT=8765
HOST="127.0.0.1"
OPEN_VIEWER=1
SERVE_SECONDS=""
TRAIN_ITERS=20
PROMPT_ROWS=4
SAMPLE_FRAMES=100
OUT_DIR="${REPO_ROOT}/out/conway_life_demo"
DEVICE="cpu"
DTYPE="float32"
BLOCK_SIZE=4
BATCH_SIZE=4
DATASET_ARGS=()

usage() {
  cat <<EOF
Usage: $0 [demo options] [-- get_dataset.sh options]

Demo options:
  --port N             Local HTTP server port (default: 8765)
  --host HOST          Local HTTP server host (default: 127.0.0.1)
  --no-open            Print the viewer URL but do not launch a browser
  --serve-seconds N    Stop automatically after N seconds (useful in CI)
  --train-iters N      Tiny demo training iterations before sampling (default: 20)
  --prompt-rows N      Validation rows used as the sampling prompt (default: 4)
  --sample-frames N    New frames to sample and append after validation (default: 100)
  --sample-rows N      Alias for --sample-frames
  --out-dir PATH       Training/sample output directory (default: out/conway_life_demo)
  --device DEVICE      Torch device for train/sample (default: cpu)
  --dtype DTYPE        Torch dtype for train/sample (default: float32)
  --block-size N       Model context length for the tiny demo (default: 4)
  --batch-size N       Training batch size for the tiny demo (default: 4)
  -h, --help           Show this help

Any arguments after -- are passed to data/conway_life_mc_int/get_dataset.sh.
Examples:
  $0
  $0 --no-open --serve-seconds 5 --train-iters 2 --sample-frames 12
  $0 -- --width 16 --height 16 --episodes 8 --steps 32 --seed 2026
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --no-open) OPEN_VIEWER=0; shift ;;
    --serve-seconds) SERVE_SECONDS="$2"; shift 2 ;;
    --train-iters) TRAIN_ITERS="$2"; shift 2 ;;
    --prompt-rows) PROMPT_ROWS="$2"; shift 2 ;;
    --sample-frames|--sample-rows) SAMPLE_FRAMES="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --block-size) BLOCK_SIZE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --) shift; DATASET_ARGS+=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown demo option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

cd "${REPO_ROOT}"
mkdir -p "${OUT_DIR}"

echo "[preflight] Checking Python training dependencies..."
python3 - <<'PY'
missing = []
for module in ("torch", "rich", "numpy"):
    try:
        __import__(module)
    except ImportError:
        missing.append(module)
if missing:
    raise SystemExit(
        "Missing Python modules needed for train/sample: "
        + ", ".join(missing)
        + ". Install repo dependencies first, for example: pip install -r requirements_cpu.txt"
    )
PY

OUTPUT_ROOT="conway_life_mc_int"
INPUT_CSV="${SCRIPT_DIR}/input.csv"
for ((i = 0; i < ${#DATASET_ARGS[@]}; i++)); do
  case "${DATASET_ARGS[$i]}" in
    --output_root) OUTPUT_ROOT="${DATASET_ARGS[$((i + 1))]}" ;;
    --output_csv) INPUT_CSV="${DATASET_ARGS[$((i + 1))]}" ;;
  esac
done
MANIFEST_PATH="${REPO_ROOT}/data/${OUTPUT_ROOT}/manifest.json"
PROMPT_CSV="${OUT_DIR}/validation_prompt.csv"
VALIDATION_CSV="${OUT_DIR}/validation.csv"
EFFECTIVE_BLOCK_SIZE="${BLOCK_SIZE}"
PROMPT_ROWS_EFFECTIVE="${PROMPT_ROWS}"
SAMPLE_START_FRAME=""
SAMPLE_CONTINUATION_CSV="${OUT_DIR}/sample_continuation.csv"
VIEWER_CSV="${OUT_DIR}/validation_plus_sample.csv"

echo "[1/7] Generating Conway Life CSV and multicontext dataset..."
"${SCRIPT_DIR}/get_dataset.sh" "${DATASET_ARGS[@]}"

echo "[2/7] Preparing validation prompt CSV from the validation split..."
PREP_OUTPUT=$(MANIFEST_PATH="${MANIFEST_PATH}" PROMPT_CSV="${PROMPT_CSV}" VALIDATION_CSV="${VALIDATION_CSV}" PROMPT_ROWS="${PROMPT_ROWS}" REQUESTED_BLOCK_SIZE="${BLOCK_SIZE}" python3 - <<'PY'
import csv
import json
import os
from pathlib import Path

manifest_path = Path(os.environ["MANIFEST_PATH"])
prompt_path = Path(os.environ["PROMPT_CSV"])
validation_path = Path(os.environ["VALIDATION_CSV"])
prompt_rows = int(os.environ["PROMPT_ROWS"])
manifest = json.loads(manifest_path.read_text())
source_csv = Path(manifest["source_csv"])
train_rows = int(manifest["train_rows"])

with source_csv.open(newline="", encoding="utf-8") as f:
    rows = list(csv.reader(f))
header, data_rows = rows[0], rows[1:]
validation_rows = data_rows[train_rows:]
if len(validation_rows) < 1:
    raise SystemExit("Validation split is empty; increase generated rows or lower train_ratio.")
prompt_rows = max(1, min(prompt_rows, len(validation_rows)))

for path, selected in ((validation_path, validation_rows), (prompt_path, validation_rows[:prompt_rows])):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(selected)
requested_block_size = int(os.environ["REQUESTED_BLOCK_SIZE"])
max_block_size = min(len(data_rows[:train_rows]) - 1, len(validation_rows) - 1)
if max_block_size < 1:
    raise SystemExit("Need at least two train rows and two validation rows for the demo training block.")
effective_block_size = max(1, min(requested_block_size, max_block_size))
print(f"validation rows: {len(validation_rows)}; prompt rows: {prompt_rows}; block_size: {effective_block_size}")
print(f"EFFECTIVE_BLOCK_SIZE={effective_block_size}")
print(f"PROMPT_ROWS_EFFECTIVE={prompt_rows}")
PY
)
echo "${PREP_OUTPUT}" | sed '/^EFFECTIVE_BLOCK_SIZE=/d' | sed '/^PROMPT_ROWS_EFFECTIVE=/d'
EFFECTIVE_BLOCK_SIZE="$(echo "${PREP_OUTPUT}" | awk -F= '/^EFFECTIVE_BLOCK_SIZE=/{print $2}' | tail -n 1)"
PROMPT_ROWS_EFFECTIVE="$(echo "${PREP_OUTPUT}" | awk -F= '/^PROMPT_ROWS_EFFECTIVE=/{print $2}' | tail -n 1)"
if [[ -z "${EFFECTIVE_BLOCK_SIZE}" || -z "${PROMPT_ROWS_EFFECTIVE}" ]]; then
  echo "Could not determine effective block size or prompt row count" >&2
  exit 1
fi

if [[ "${EFFECTIVE_BLOCK_SIZE}" != "${BLOCK_SIZE}" ]]; then
  echo "Requested block_size=${BLOCK_SIZE} is too large for the validation split; using block_size=${EFFECTIVE_BLOCK_SIZE}."
fi

echo "[3/7] Validating generated manifest and collecting context datasets..."
mapfile -t MULTICONTEXT_DATASETS < <(MANIFEST_PATH="${MANIFEST_PATH}" python3 - <<'PY'
import json
import os
import re
from pathlib import Path

manifest = json.loads(Path(os.environ["MANIFEST_PATH"]).read_text())
columns = {col["source_column"]: col for col in manifest["columns"]}
pixel_contexts = [name for name in manifest["multicontext_datasets"] if re.fullmatch(r".*/p\d+", name)]
width = columns["width"]["actual_int_min"]
height = columns["height"]["actual_int_min"]
expected_pixels = width * height
assert manifest["rows"] > 1, "manifest must have at least two rows"
assert columns["width"]["actual_int_min"] == columns["width"]["actual_int_max"], "width must be fixed"
assert columns["height"]["actual_int_min"] == columns["height"]["actual_int_max"], "height must be fixed"
assert len(pixel_contexts) == expected_pixels, f"expected {expected_pixels} p* contexts, got {len(pixel_contexts)}"
print("\n".join(manifest["multicontext_datasets"]))
PY
)
echo "manifest ok: ${#MULTICONTEXT_DATASETS[@]} contexts"

echo "[4/7] Training a tiny multicontext model before sampling..."
python3 train.py \
  --training_mode multicontext \
  --multicontext \
  --multicontext_datasets "${MULTICONTEXT_DATASETS[@]}" \
  --out_dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --block_size "${EFFECTIVE_BLOCK_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --n_layer 2 \
  --n_head 2 \
  --n_embd 64 \
  --max_iters "${TRAIN_ITERS}" \
  --eval_interval "${TRAIN_ITERS}" \
  --eval_iters 1 \
  --learning_rate 1e-3 \
  --always_save_checkpoint \
  --no-compile \
  --no-tensorboard_log \
  --no-wandb_log

echo "[5/7] Sampling from validation start tokens..."
python3 sample.py \
  --init_from resume \
  --out_dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --num_samples 1 \
  --max_new_tokens "${SAMPLE_FRAMES}" \
  --temperature 0.8 \
  --top_k 5 \
  --no-compile \
  --multicontext \
  --multicontext_datasets "${MULTICONTEXT_DATASETS[@]}" \
  --multicontext_csv_input "${PROMPT_CSV}" \
  --multicontext_csv_output_file "${SAMPLE_CONTINUATION_CSV}" \
  --no-multicontext_csv_output_include_prompt

echo "[6/7] Appending sampled rows to the validation CSV for viewing..."
COMBINE_OUTPUT=$(VALIDATION_CSV="${VALIDATION_CSV}" SAMPLE_CONTINUATION_CSV="${SAMPLE_CONTINUATION_CSV}" VIEWER_CSV="${VIEWER_CSV}" python3 - <<'PY'
import csv
import os
from pathlib import Path

validation_path = Path(os.environ["VALIDATION_CSV"])
sample_path = Path(os.environ["SAMPLE_CONTINUATION_CSV"])
viewer_path = Path(os.environ["VIEWER_CSV"])

with validation_path.open(newline="", encoding="utf-8") as f:
    validation_rows = list(csv.reader(f))
with sample_path.open(newline="", encoding="utf-8") as f:
    sample_rows = list(csv.reader(f))
if not validation_rows or not sample_rows:
    raise SystemExit("Validation/sample CSV is empty")
if validation_rows[0] != sample_rows[0]:
    raise SystemExit("Validation and sample CSV headers do not match")
viewer_path.parent.mkdir(parents=True, exist_ok=True)
with viewer_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(validation_rows)
    writer.writerows(sample_rows[1:])
sample_start_frame = len(validation_rows) - 1
print(f"wrote {viewer_path} with {sample_start_frame} validation rows + {len(sample_rows) - 1} sampled rows")
print(f"SAMPLE_START_FRAME={sample_start_frame}")
PY
)
echo "${COMBINE_OUTPUT}" | sed '/^SAMPLE_START_FRAME=/d'
SAMPLE_START_FRAME="$(echo "${COMBINE_OUTPUT}" | awk -F= '/^SAMPLE_START_FRAME=/{print $2}' | tail -n 1)"
if [[ -z "${SAMPLE_START_FRAME}" ]]; then
  echo "Could not determine sample start frame" >&2
  exit 1
fi

CSV_ABS="$(python3 - <<PY
from pathlib import Path
print(Path(${VIEWER_CSV@Q}).expanduser().resolve())
PY
)"
CSV_PATH="/$(python3 - <<PY
from pathlib import Path
repo = Path(${REPO_ROOT@Q}).resolve()
csv = Path(${CSV_ABS@Q}).resolve()
try:
    print(csv.relative_to(repo).as_posix())
except ValueError as exc:
    raise SystemExit(f"Demo CSV must be inside the repository so the local server can serve it: {csv}") from exc
PY
)"
VIEWER_PATH="/data/roomba/roomba_grayscale_viewer.html"
VIEWER_URL="http://${HOST}:${PORT}${VIEWER_PATH}?csv=${CSV_PATH}&prompt_rows=${PROMPT_ROWS_EFFECTIVE}&sample_start_frame=${SAMPLE_START_FRAME}"

echo "[7/7] Starting local HTTP server at http://${HOST}:${PORT}/ ..."
python3 -m http.server "${PORT}" --bind "${HOST}" --directory "${REPO_ROOT}" >/tmp/conway_life_viewer_server.log 2>&1 &
SERVER_PID=$!
cleanup() {
  if kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
sleep 1
if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
  echo "HTTP server failed to start. Log:" >&2
  cat /tmp/conway_life_viewer_server.log >&2
  exit 1
fi

echo "Viewer URL: ${VIEWER_URL}"
if [[ "${OPEN_VIEWER}" -eq 1 ]]; then
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${VIEWER_URL}" >/dev/null 2>&1 || echo "Could not open browser automatically; paste the URL above into a browser."
  elif command -v open >/dev/null 2>&1; then
    open "${VIEWER_URL}" >/dev/null 2>&1 || echo "Could not open browser automatically; paste the URL above into a browser."
  else
    VIEWER_URL="${VIEWER_URL}" python3 - <<'PY' || echo "Could not open browser automatically; paste the URL above into a browser."
import os
import webbrowser
webbrowser.open(os.environ["VIEWER_URL"])
PY
  fi
else
  echo "--no-open set; paste the URL above into a browser."
fi

if [[ -n "${SERVE_SECONDS}" ]]; then
  echo "Serving for ${SERVE_SECONDS} seconds..."
  sleep "${SERVE_SECONDS}"
else
  echo "Serving viewer until interrupted. Press Ctrl+C to stop."
  wait "${SERVER_PID}"
fi
