#!/bin/bash
# run_all_code_annotations.sh

set -u

# Color codes
BLUE='\033[0;34m'
GREEN='\033[0;32m'
MAGENTA='\033[0;35m'
RED='\033[0;31m'
NC='\033[0m'

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <python_file>" >&2
  exit 1
fi

# filename to convert
filename="$1"
file_dir="$(cd "$(dirname "$filename")" && pwd)"
file_base="$(basename "$filename")"
parent_dir="$(cd "${file_dir}/.." && pwd)"
mapped_file="${filename}.mapped"

# List of every supported mode
MODES=(
  general exact keywords nesting param_nesting argnum
  dot_nesting name_kind literals semantic comments scope
)

failures=0

for mode in "${MODES[@]}"; do
  echo -e "${MAGENTA}=== MODE: $mode ===${NC}"
  echo

  if ! python code_highlighter.py --mode "$mode" "$filename" 2>&1; then
    echo -e "${RED}Highlighter failed for mode $mode${NC}"
    ((failures+=1))
    echo
    continue
  fi

  mode_dir="${parent_dir}/${mode}_annotations"
  if [[ ! -d "$mode_dir" ]]; then
    mkdir -p "$mode_dir"
  fi

  target_mapped="${mode_dir}/${file_base%.py}.mapped"
  if ! mv "$mapped_file" "$target_mapped"; then
    echo -e "${RED}Failed to move mapped output for mode $mode${NC}"
    ((failures+=1))
    echo
    continue
  fi
  echo

  echo -e "${GREEN}--- ${target_mapped} (mapped output) ---${NC}"
  if ! cat "$target_mapped"; then
    echo -e "${RED}Unable to read mapped output for mode $mode${NC}"
    ((failures+=1))
  fi
  echo

  echo -e "${BLUE}--- $filename (original source) ---${NC}"
  if ! cat "$filename"; then
    echo -e "${RED}Unable to read original source${NC}"
    ((failures+=1))
  fi
  echo
done

if (( failures > 0 )); then
  echo -e "${RED}${failures} mode(s) reported issues.${NC}" >&2
  exit 1
fi

