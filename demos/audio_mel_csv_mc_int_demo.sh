#!/usr/bin/env bash
# Audio mel regular integer CSV multicontext demo:
# 1) convert audio to quantized Whisper mel CSV columns
# 2) split columns into per-channel integer datasets via data/csv_mc_int
# 3) train regular multicontext
# 4) sample, reconstruct WAVs including prompt/start tokens, and write Plotly HTML
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
AUDIO_INPUT="${1:-data/audio_mel_csv/demo_audio}"
OUTPUT_ROOT="${AUDIO_MEL_OUTPUT_ROOT:-audio_mel_csv}"
WORK_DIR="${AUDIO_MEL_WORK_DIR:-data/audio_mel_csv/work}"
OUT_DIR="${AUDIO_MEL_OUT_DIR:-out/audio_mel_csv}"
MAX_ITERS="${AUDIO_MEL_MAX_ITERS:-1000}"
DEVICE="${AUDIO_MEL_DEVICE:-cuda:0}"
DTYPE="${AUDIO_MEL_DTYPE:-bfloat16}"

find_audio_files() {
  local path="$1"
  if [[ -f "$path" ]]; then
    printf '%s\n' "$path"
  elif [[ -d "$path" ]]; then
    find "$path" -type f \( -iname "*.wav" -o -iname "*.wave" -o -iname "*.mp3" -o -iname "*.flac" -o -iname "*.ogg" -o -iname "*.m4a" \) -print
  fi
}

create_demo_audio_folder() {
  local folder="$1"
  if ! command -v sox >/dev/null 2>&1; then
    echo "No audio files found at '$folder' and sox is not installed. Install sox or pass an audio file/folder." >&2
    exit 1
  fi
  mkdir -p "$folder"
  echo "No audio files found at '$folder'; generating demo WAV files with sox."
  sox -n -r 16000 -c 1 "$folder/000_sine_220hz.wav" synth 1.5 sine 220 fade 0.02 1.5 0.02 gain -6
  sox -n -r 16000 -c 1 "$folder/001_sine_330hz.wav" synth 1.5 sine 330 fade 0.02 1.5 0.02 gain -6
  sox -n -r 16000 -c 1 "$folder/002_chord.wav" synth 1.5 sine 261.63 sine 329.63 sine 392.00 fade 0.02 1.5 0.02 gain -10
}

if [[ -z "$(find_audio_files "$AUDIO_INPUT" | head -n 1)" ]]; then
  if [[ -f "$AUDIO_INPUT" ]]; then
    echo "'$AUDIO_INPUT' exists but is not a supported audio file (.wav/.wave/.mp3/.flac/.ogg/.m4a)." >&2
    exit 1
  fi
  create_demo_audio_folder "$AUDIO_INPUT"
fi

python3 data/audio_mel_csv/prepare_audio_mel_csv.py "$AUDIO_INPUT" \
  --output_root "$OUTPUT_ROOT" \
  --work_dir "$WORK_DIR" \
  --train_ratio "${AUDIO_MEL_TRAIN_RATIO:-0.9}"

mapfile -t DATASETS < <(python3 - <<PY
import json
from pathlib import Path
manifest = json.loads(Path('data/$OUTPUT_ROOT/manifest.json').read_text())
for dataset in manifest['multicontext_datasets']:
    print(dataset)
PY
)

python3 train.py \
  --training_mode multicontext \
  --dataset "${DATASETS[0]}" \
  --multicontext \
  --multicontext_datasets "${DATASETS[@]}" \
  --n_layer 6 \
  --n_head 6 \
  --attention_variant infinite \
  --use_concat_heads \
  --n_qk_head_dim 200 \
  --n_v_head_dim 112 \
  --n_embd 128 \
  --block_size 100 \
  --batch_size 2 \
  --max_iters "$MAX_ITERS" \
  --eval_interval 50 \
  --eval_iters 10 \
  --learning_rate 1e-3 \
  --dropout 0.0 \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --optimizer muon \
  --weight_decay 0.0 \
  --softmax_variant_attn relu2max \
  --use_qk_norm \
  --use_qk_norm_scale \
  --use_rotary_embeddings \
  --no-use_abs_pos_embeddings \
  --compile \
  --out_dir "$OUT_DIR"

read -r -a VIEWER_SEEDS <<<"${AUDIO_MEL_VIEWER_SEEDS:-1337 1338 1339}"
read -r -a VIEWER_TOP_K <<<"${AUDIO_MEL_VIEWER_TOP_K:-1 5}"
python3 data/audio_mel_csv/generate_audio_mel_comparison.py \
  --input_csv "$WORK_DIR/mel_int.csv" \
  --manifest "data/$OUTPUT_ROOT/manifest.json" \
  --checkpoint_dir "$OUT_DIR" \
  --work_dir "${AUDIO_MEL_VIEWER_WORK_DIR:-$OUT_DIR/audio_viewer}" \
  --holdout_rows "${AUDIO_MEL_VIEWER_HOLDOUT_ROWS:-128}" \
  --prompt_rows "${AUDIO_MEL_VIEWER_PROMPT_ROWS:-512}" \
  --seeds "${VIEWER_SEEDS[@]}" \
  --top_k "${VIEWER_TOP_K[@]}" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --compile
