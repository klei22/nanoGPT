#!/bin/bash
# demos/dataset_analysis.sh

set -euo pipefail

output_dir="${1:?Usage: $0 <out_dir> [dataset]}"
dataset="${2:-minipile}"

run_name="$(basename "$output_dir")"
results_dir="results"
mkdir -p "$results_dir"

python analyze_with_dataset.py \
  --out_dir "$output_dir" \
  --dataset "$dataset" \
  --display topk \
  --activation_heatmap \
  --activation_heatmap_file "$results_dir/${run_name}_activation_heatmap.html" \
  --activation_hist_file "$results_dir/${run_name}_activation_hist.html" \
  --attention_head_heatmap_file "$results_dir/${run_name}_attn_head_heatmap.html" \
  --attention_head_hist_file "$results_dir/${run_name}_attn_head_hist.html"
