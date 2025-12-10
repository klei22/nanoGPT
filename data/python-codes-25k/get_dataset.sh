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
# 7. Combine --outputs-only with --run-annotations to execute all code-highlighter modes and record any failures.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add url with dataset here:
url="https://huggingface.co/datasets/flytech/python-codes-25k/tree/main"

annotate=false

if [[ "${1:-}" == "--outputs-only" ]]; then
  shift

  target_dir="${SCRIPT_DIR}/outputs"
  raw_output_file="${SCRIPT_DIR}/output_only.txt"

  target_dir_set=false
  raw_output_file_set=false

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run-annotations)
        annotate=true
        shift
        ;;
      *)
        if [[ "${target_dir_set}" == false ]]; then
          target_dir="$1"
          target_dir_set=true
          shift
        elif [[ "${raw_output_file_set}" == false ]]; then
          raw_output_file="$1"
          raw_output_file_set=true
          shift
        else
          echo "Unexpected argument: $1" >&2
          exit 1
        fi
        ;;
    esac
  done

  python3 "${SCRIPT_DIR}/../template/utils/get_json_dataset.py" \
    --url "${url}" \
    --include_keys "instruction" "output" \
    --value_prefix $'\n"""<start>\n' $'\n"""\n' \
    --skip_empty \
    --output_text_file "${raw_output_file}"

  python3 "${SCRIPT_DIR}/split_outputs.py" \
    --input "${raw_output_file}" \
    --format \
    --output_dir "${target_dir}"

  if [[ "${annotate}" == true ]]; then
    target_dir_abs="$(cd "${target_dir}" && pwd)"
    annotation_log="${target_dir_abs}/../annotation_results.log"
    echo "Running code annotations for files in ${target_dir_abs}" | tee "${annotation_log}"

    pushd "${SCRIPT_DIR}" >/dev/null
    failures=0
    for py_file in "${target_dir_abs}"/*.py; do
      [[ -e "${py_file}" ]] || continue
      echo "Annotating ${py_file}" | tee -a "${annotation_log}"
      if bash "${SCRIPT_DIR}/run_all_code_annotations.sh" "${py_file}" >>"${annotation_log}" 2>&1; then
        echo "OK: ${py_file}" | tee -a "${annotation_log}"
      else
        echo "FAIL: ${py_file}" | tee -a "${annotation_log}"
        ((failures+=1))
      fi
      echo | tee -a "${annotation_log}"
    done
    popd >/dev/null

    if (( failures > 0 )); then
      echo "Annotation runs completed with ${failures} failure(s). See ${annotation_log} for details." | tee -a "${annotation_log}" >&2
    else
      echo "All annotation runs completed successfully. See ${annotation_log} for details." | tee -a "${annotation_log}"
    fi
  fi

  exit 0
fi

# uncomment and fill in if url has json datasets
# Note: the $'\n' syntax allows for special characters like \n
python3 "${SCRIPT_DIR}/../template/utils/get_json_dataset.py" \
  --url "${url}" \
  --include_keys "instruction" "input" "output" \
  --value_prefix $'#U:\n' $'#B:\n' $'\n'
