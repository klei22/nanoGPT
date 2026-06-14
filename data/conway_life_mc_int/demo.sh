#!/usr/bin/env bash
# End-to-end Conway Life dataset demo:
#   1. generate the CSV and multicontext manifest
#   2. validate key manifest fields
#   3. serve the repo over localhost
#   4. open the viewer preloaded with the generated CSV

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PORT=8765
HOST="127.0.0.1"
OPEN_VIEWER=1
SERVE_SECONDS=""
DATASET_ARGS=()

usage() {
  cat <<EOF
Usage: $0 [demo options] [-- get_dataset.sh options]

Demo options:
  --port N             Local HTTP server port (default: 8765)
  --host HOST          Local HTTP server host (default: 127.0.0.1)
  --no-open            Print the viewer URL but do not launch a browser
  --serve-seconds N    Stop automatically after N seconds (useful in CI)
  -h, --help           Show this help

Any arguments after -- are passed to data/conway_life_mc_int/get_dataset.sh.
Examples:
  $0
  $0 --serve-seconds 5 --no-open
  $0 -- --width 16 --height 16 --episodes 8 --steps 32 --seed 2026
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --no-open) OPEN_VIEWER=0; shift ;;
    --serve-seconds) SERVE_SECONDS="$2"; shift 2 ;;
    --) shift; DATASET_ARGS+=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown demo option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

cd "${REPO_ROOT}"

OUTPUT_ROOT="conway_life_mc_int"
INPUT_CSV="${SCRIPT_DIR}/input.csv"
for ((i = 0; i < ${#DATASET_ARGS[@]}; i++)); do
  case "${DATASET_ARGS[$i]}" in
    --output_root) OUTPUT_ROOT="${DATASET_ARGS[$((i + 1))]}" ;;
    --output_csv) INPUT_CSV="${DATASET_ARGS[$((i + 1))]}" ;;
  esac
done
MANIFEST_PATH="${REPO_ROOT}/data/${OUTPUT_ROOT}/manifest.json"
CSV_ABS="$(python3 - <<PY
from pathlib import Path
print(Path(${INPUT_CSV@Q}).expanduser().resolve())
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

echo "[1/4] Generating Conway Life CSV and multicontext dataset..."
"${SCRIPT_DIR}/get_dataset.sh" "${DATASET_ARGS[@]}"

echo "[2/4] Validating generated manifest..."
MANIFEST_PATH="${MANIFEST_PATH}" python3 - <<'PY'
import json
import os
import re
from pathlib import Path

manifest_path = Path(os.environ["MANIFEST_PATH"])
manifest = json.loads(manifest_path.read_text())
columns = {col["source_column"]: col for col in manifest["columns"]}
pixel_contexts = [name for name in manifest["multicontext_datasets"] if re.fullmatch(r".*/p\d+", name)]
width = columns["width"]["actual_int_min"]
height = columns["height"]["actual_int_min"]
expected_pixels = width * height
assert manifest["rows"] > 1, "manifest must have at least two rows"
assert columns["width"]["actual_int_min"] == columns["width"]["actual_int_max"], "width must be fixed"
assert columns["height"]["actual_int_min"] == columns["height"]["actual_int_max"], "height must be fixed"
assert len(pixel_contexts) == expected_pixels, f"expected {expected_pixels} p* contexts, got {len(pixel_contexts)}"
print(f"manifest ok: {manifest['rows']} rows, {width}x{height}, {len(pixel_contexts)} pixel contexts")
PY

VIEWER_PATH="/data/roomba/roomba_grayscale_viewer.html"
VIEWER_URL="http://${HOST}:${PORT}${VIEWER_PATH}?csv=${CSV_PATH}"

echo "[3/4] Starting local HTTP server at http://${HOST}:${PORT}/ ..."
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

echo "[4/4] Viewer URL: ${VIEWER_URL}"
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
