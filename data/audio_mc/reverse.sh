#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Reverse a Whisper-style mel CSV back into a WAV using the parameters
# stored in meta.pkl (written by prepare.py --method whisper_mel_csv).
#
# Usage:
#   bash reverse_latest_csv_to_wav.sh
#   bash reverse_latest_csv_to_wav.sh mel.csv
#   bash reverse_latest_csv_to_wav.sh mel.csv out.wav
#   bash reverse_latest_csv_to_wav.sh mel.csv out.wav 64   # Griffin-Lim iters
#
# Notes:
# - Reconstruction from mel is inherently lossy (no phase) and uses Griffin-Lim.
# - Increase Griffin-Lim iterations for better quality (slower).

CSV_PATH="${1:-}"
OUT_WAV="${2:-}"
GRIFFIN_LIM_ITERS="${3:-32}"

if [[ -z "${CSV_PATH}" ]]; then
  # Pick most recently modified CSV in current dir.
  # (ls -t sorts newest first)
  if compgen -G "*.csv" > /dev/null; then
    CSV_PATH="$(ls -t *.csv | head -n 1)"
  else
    echo "ERROR: No .csv files found in $(pwd). Provide a CSV path as the first argument."
    exit 1
  fi
fi

if [[ ! -f "${CSV_PATH}" ]]; then
  echo "ERROR: CSV file not found: ${CSV_PATH}"
  exit 1
fi

if [[ -z "${OUT_WAV}" ]]; then
  base="${CSV_PATH%.*}"
  OUT_WAV="${base}_reconstructed.wav"
fi

if [[ ! -f "mel_csv_to_wav.py" ]]; then
  echo "ERROR: mel_csv_to_wav.py not found in $(pwd)."
  echo "       Put this script next to mel_csv_to_wav.py or adjust the path in the script."
  exit 1
fi

# Pull parameters from meta.pkl if present and looks like whisper_mel_csv meta.
META_ARGS=""
if [[ -f "meta.pkl" ]]; then
  META_ARGS="$(python3 - <<'PY'
import pickle
import shlex
from pathlib import Path

meta_path = Path("meta.pkl")
try:
    meta = pickle.loads(meta_path.read_bytes())
except Exception:
    print("")  # ignore meta issues
    raise SystemExit(0)

if meta.get("tokenizer") != "whisper_mel_csv":
    print("")  # not the meta we want
    raise SystemExit(0)

def pick(key, default):
    v = meta.get(key, default)
    return default if v is None else v

args = []
# Match mel_csv_to_wav.py flags
args += ["--sample_rate", str(pick("sample_rate", 16000))]
args += ["--n_fft", str(pick("n_fft", 400))]
args += ["--hop_length", str(pick("hop_length", 160))]
args += ["--win_length", str(pick("win_length", 400))]
args += ["--n_mels", str(pick("n_mels", 80))]
args += ["--f_min", str(pick("f_min", 0.0))]
args += ["--f_max", str(pick("f_max", 8000.0))]
args += ["--power", str(pick("power", 2.0))]

# Booleans: your mel_csv_to_wav.py uses BooleanOptionalAction (True by default)
center = bool(pick("center", True))
normalize = bool(pick("normalize", True))

args += ["--center" if center else "--no-center"]
args += ["--normalized_input" if normalize else "--no-normalized_input"]

# Keep mel filterbank conventions consistent with your forward transform
# (prepare.py/WhisperMelCsvTokenizer uses mel_scale="slaney", norm="slaney")
args += ["--mel_scale", "slaney", "--mel_norm", "slaney"]

print(" ".join(shlex.quote(x) for x in args))
PY
)"
fi

echo "[reverse] CSV: ${CSV_PATH}"
echo "[reverse] OUT: ${OUT_WAV}"
if [[ -n "${META_ARGS}" ]]; then
  echo "[reverse] Using meta.pkl params."
else
  echo "[reverse] meta.pkl missing/unusable; using mel_csv_to_wav.py defaults."
fi

# shellcheck disable=SC2086
python3 mel_csv_to_wav.py "${CSV_PATH}" \
  --output "${OUT_WAV}" \
  --griffin_lim_iters "${GRIFFIN_LIM_ITERS}" \
  ${META_ARGS}

echo "[reverse] Done: ${OUT_WAV}"

