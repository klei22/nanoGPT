#!/usr/bin/env bash
set -euo pipefail

# Precompile the recursive-angle webapp's existing CSV exports into a fast
# single-canvas HTML viewer that keeps the same dark rectangular node/edge look.
#
# Usage with already exported files:
#   bash demos/min_angle_graph_fast_viewer_demo.sh \
#     out/recursive_group_123_adjacency.csv \
#     out/recursive_group_token_list.csv \
#     out/recursive_group_123_dictionary.json

adjacency_csv="${1:-}"
token_list_csv="${2:-}"
dictionary_json="${3:-}"

if [[ -n "${adjacency_csv}" ]]; then
  cmd=(
    python3 demos/min_angle_graph_fast_viewer_demo.py
    --adjacency-csv "${adjacency_csv}"
    --output-html out/min_angle_graph_fast_viewer.html
  )
  [[ -n "${token_list_csv}" ]] && cmd+=(--token-list-csv "${token_list_csv}")
  [[ -n "${dictionary_json}" ]] && cmd+=(--dictionary-json "${dictionary_json}")
  "${cmd[@]}"
else
  echo "No adjacency CSV supplied; generating a synthetic demo graph." >&2
  python3 demos/min_angle_graph_fast_viewer_demo.py \
    --nodes 300 \
    --edges-per-node 4 \
    --frames 120 \
    --output-html out/min_angle_graph_fast_viewer.html
fi

# Optional offline/video preprocessing path for the same precompiled viewer:
# python3 demos/min_angle_graph_fast_viewer_demo.py \
#   --adjacency-csv out/recursive_group_123_adjacency.csv \
#   --token-list-csv out/recursive_group_token_list.csv \
#   --dictionary-json out/recursive_group_123_dictionary.json \
#   --output-html out/min_angle_graph_fast_viewer_video.html \
#   --frame-dir out/min_angle_graph_frames \
#   --video-out out/min_angle_graph_fast_viewer.webm \
#   --workers "$(python3 - <<'PY'
# import os
# print(max(1, (os.cpu_count() or 2) // 2))
# PY
# )"
