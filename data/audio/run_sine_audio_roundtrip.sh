#!/usr/bin/env bash
# End-to-end toy audio demo:
# 1) create dummy sine-wave WAV
# 2) optionally play it
# 3) encode to whisper-style mel CSV
# 4) decode CSV back to WAV
# 5) optionally play recovered audio
# 6) create a headered CSV for int multicontext demos

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ORIG_WAV="${1:-dummy_sine.wav}"
MEL_CSV="${2:-dummy_sine_mel.csv}"
RECOVERED_WAV="${3:-dummy_sine_recovered.wav}"
NUMINT_CSV="${4:-audio_num_int_input.csv}"

python3 generate_sine_wav.py --output "$ORIG_WAV"

play_audio_if_available() {
  local wav_path="$1"
  if command -v ffplay >/dev/null 2>&1; then
    ffplay -hide_banner -loglevel error -autoexit -nodisp "$wav_path"
  elif command -v aplay >/dev/null 2>&1; then
    aplay "$wav_path"
  elif command -v play >/dev/null 2>&1; then
    play -q "$wav_path"
  else
    echo "No local audio player (ffplay/aplay/play) found; skipping playback for $wav_path"
  fi
}

play_audio_if_available "$ORIG_WAV"

python3 prepare.py \
  --method whisper_mel_csv \
  --train_input "$ORIG_WAV" \
  --train_output "$MEL_CSV" \
  --percentage_train 1.0

python3 ../template/mel_csv_to_wav.py "$MEL_CSV" --output "$RECOVERED_WAV"
play_audio_if_available "$RECOVERED_WAV"

python3 mel_csv_to_numint_csv.py \
  --input_csv "$MEL_CSV" \
  --output_csv "$NUMINT_CSV" \
  --bins "10,30,60"

cat <<EOF
Roundtrip complete:
  original wav:   $SCRIPT_DIR/$ORIG_WAV
  mel csv:        $SCRIPT_DIR/$MEL_CSV
  recovered wav:  $SCRIPT_DIR/$RECOVERED_WAV
  num-int csv:    $SCRIPT_DIR/$NUMINT_CSV
EOF
