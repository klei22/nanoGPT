#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/orig"

python3 "${SCRIPT_DIR}/prepare.py" \
  --output-dir "${OUTPUT_DIR}"
