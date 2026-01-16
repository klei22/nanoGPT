#!/bin/bash
set -euo pipefail

# Download Kaikki.org dictionary JSONL dumps for selected languages.
# Usage examples:
#   bash get_dataset.sh                # downloads English by default
#   bash get_dataset.sh Korean         # downloads a single language
#   bash get_dataset.sh English Korean # downloads multiple languages
#   bash get_dataset.sh all            # downloads every language listed below
#
# The script fetches the "All word forms" dumps published at
# https://kaikki.org for each selected language.

AVAILABLE_LANGS=(
  Arabic
  Chinese
  English
  French
  German
  Hindi
  Indonesian
  Italian
  Japanese
  Korean
  Portuguese
  Russian
  Spanish
  Vietnamese
)

DEFAULT_LANG="English"
BASE_URL="https://kaikki.org/dictionary"

usage() {
  cat <<'USAGE'
Download Kaikki.org dictionary JSONL files for one or more languages.

Options:
  all         Download every language in the curated list.
  <language>  Download a specific language from the list above. You can
              pass multiple language names separated by spaces.

Examples:
  bash get_dataset.sh
  bash get_dataset.sh Korean
  bash get_dataset.sh English Korean
  bash get_dataset.sh all
USAGE
}

if [[ ${#} -eq 0 ]]; then
  selection=("${DEFAULT_LANG}")
elif [[ ${1} == "-h" || ${1} == "--help" ]]; then
  usage
  exit 0
elif [[ ${1} == "all" ]]; then
  selection=("${AVAILABLE_LANGS[@]}")
else
  selection=("${@}")
fi

# Verify selection
for lang in "${selection[@]}"; do
  if [[ ! " ${AVAILABLE_LANGS[*]} " =~ (^|[[:space:]])${lang}($|[[:space:]]) ]]; then
    echo "[ERROR] Unsupported language: ${lang}" >&2
    echo "Supported languages: ${AVAILABLE_LANGS[*]}" >&2
    exit 1
  fi
  dest_file="kaikki.org-dictionary-${lang}.jsonl"
  url="${BASE_URL}/${lang}/kaikki.org-dictionary-${lang}.jsonl"
  echo "Downloading ${lang} dictionary from ${url}" >&2
  curl -L --fail --progress-bar -o "${dest_file}" "${url}"
done
