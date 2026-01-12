#!/bin/bash

# Download the Linux commands dataset from Hugging Face and emit prompt/command pairs
# to input.txt. Pass a language code as the first argument (default: eng).
# Available languages: ar, ch, da, de, eng, es, fr, ja, ko, pt, ru

set -euo pipefail

language="${1:-eng}"

case "${language}" in
  en)
    language="eng"
    ;;
  ar|ch|da|de|eng|es|fr|ja|ko|pt|ru)
    ;;
  *)
    echo "Unsupported language: ${language}" >&2
    echo "Usage: $0 [ar|ch|da|de|eng|es|fr|ja|ko|pt|ru]" >&2
    exit 1
    ;;
 esac

url="https://huggingface.co/datasets/missvector/linux-commands/tree/main/${language}"

python3 ./utils/get_parquet_dataset.py \
  --url "${url}" \
  --include_keys "${language}" "completion" \
  --value_prefix $'#PROMPT:\n' $'#COMMAND:\n' \
  --output_text_file "input.txt" \
  --skip_empty
