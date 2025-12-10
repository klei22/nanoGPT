#!/bin/bash
set -euo pipefail

### Instructions:
# 1. Replace "INSERT_URL_WITH_FILES" with the actual URL to the Parquet files.
# 2. Modify the "include_keys" array to specify the keys you want to include in the output.
# 3. (Optionally) Modify the "value_prefixes" array to set prefixes for each value, use "" for empty prefixes
# 4. Set "--skip_empty" to true if you want to skip empty fields, or false if not needed.
# 5. Set "--no_output_text" to true if you plan to process the intermediate json files in a custom manner.
# 6. Use --outputs-only <target_dir> [raw_output_file] to emit only the "output" field and split each ```python``` block
#    into sequential files for the programming-language tokenizer.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add url with dataset here:
url="https://huggingface.co/datasets/flytech/python-codes-25k/tree/main"

if [[ "${1:-}" == "--outputs-only" ]]; then
  target_dir=${2:-"${SCRIPT_DIR}/outputs"}
  raw_output_file=${3:-"${SCRIPT_DIR}/output_only.txt"}

  python3 "${SCRIPT_DIR}/../template/utils/get_json_dataset.py" \
    --url "${url}" \
    --include_keys "instruction" "output" \
    --value_prefix $'\n"""<start>\n' $'\n"""\n' \
    --skip_empty \
    --output_text_file "${raw_output_file}"

  python3 "${SCRIPT_DIR}/split_outputs.py" \
    --input "${raw_output_file}" \
    --output_dir "${target_dir}"

  exit 0
fi

# uncomment and fill in if url has json datasets
# Note: the $'\n' syntax allows for special characters like \n
python3 "${SCRIPT_DIR}/../template/utils/get_json_dataset.py" \
  --url "${url}" \
  --include_keys "instruction" "input" "output" \
  --value_prefix $'#U:\n' $'#B:\n' $'\n'
