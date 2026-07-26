#!/usr/bin/env bash
set -euo pipefail

# 1) Run the exploration sweep comparing cross_entropy vs relu2_cross_entropy.
python3 optimization_and_search/run_experiments.py \
  --config explorations/relu2_cross_entropy_comparison.yaml \
  --config_format yaml \
  --output_dir out_relu2_ce_compare

# 2) Example sampling command using ReLU^2 probability mapping.
#    (update --out_dir to one of the run directories created above)
python3 sample.py \
  --out_dir out_relu2_ce_compare \
  --start "Once upon a time" \
  --num_samples 1 \
  --max_new_tokens 128 \
  --temperature 0.8 \
  --top_k 200 \
  --sampling_activation relu2
