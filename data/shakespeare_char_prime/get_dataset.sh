#!/usr/bin/env bash
# get_dataset.sh

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PREPARE_PY="../template/prepare.py"
CHAR_CONVERT_PY="../template/utils/char_convert.py"

# Download tiny Shakespeare dataset once.
wget -O input_raw.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Choose modulo sizes to generate.
MODULI=(2 3 5 7 11 13 17 19)

for n in "${MODULI[@]}"; do
  cp input_raw.txt input_mod.txt

  python3 "$CHAR_CONVERT_PY" input_mod.txt --method modulo_letters --modulo-n "$n"

  # Build meta.pkl from conversion alphabet, then reuse it for train/val bin creation.
  python3 "$PREPARE_PY" --method char -t tokensfile.txt -s -S "mod${n}"
  python3 "$PREPARE_PY" --method char -t input_mod.txt --reuse_chars -s -S "mod${n}"
done

rm -f input_mod.txt tokensfile.txt
