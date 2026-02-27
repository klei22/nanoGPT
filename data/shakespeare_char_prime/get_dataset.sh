#!/usr/bin/env bash
# Build several modulo-prime Shakespeare character datasets.

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SOURCE_FILE="input_source.txt"
TOKENS_FILE="tokensfile.txt"
PRIMES=(2 3 5 7 11 13 17 19 23 29)

wget -O "$SOURCE_FILE" https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

for prime in "${PRIMES[@]}"; do
  run_input="input_p${prime}.txt"
  cp "$SOURCE_FILE" "$run_input"

  python3 ./utils/char_convert.py "$run_input" --method letter_modulo --letter-modulus "$prime"

  # Build meta.pkl from the emitted token list, then tokenize the transformed data.
  python3 ./prepare.py --method char -t "$TOKENS_FILE" -s -S "p${prime}"
  python3 ./prepare.py --method char -t "$run_input" --reuse_chars -s -S "p${prime}"
done

rm -f input_p*.txt "$SOURCE_FILE"
