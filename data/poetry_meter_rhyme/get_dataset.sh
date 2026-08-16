#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
python3 download_sources.py --lock-file sources.lock.json --raw-dir _raw "$@"
python3 build_dataset.py --raw-dir _raw --output-dir generated --seed 1337
python3 validate_dataset.py --data-dir generated
cp generated/train.txt input.txt
