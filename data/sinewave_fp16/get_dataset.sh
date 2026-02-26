#!/bin/bash
set -euo pipefail

# Generate fp16-bit encoded sinewave datasets for numerical multi-context training.
python data/sinewave_fp16/create_fp16_sine_datasets.py "$@"
