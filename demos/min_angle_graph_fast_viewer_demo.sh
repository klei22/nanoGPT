#!/usr/bin/env bash
set -euo pipefail

# Fast replacement path for very large Plotly min-angle graph pages.
# The HTML keeps the full graph data but renders through one Canvas element.
# Uncomment --frame-dir/--video-out to pre-render SVG frames concurrently for video playback.

python3 demos/min_angle_graph_fast_viewer_demo.py \
  --nodes 1500 \
  --edges-per-node 4 \
  --frames 120 \
  --output-html out/min_angle_graph_fast_viewer.html

# Example offline/video preprocessing path:
# python3 demos/min_angle_graph_fast_viewer_demo.py \
#   --nodes 1500 \
#   --edges-per-node 4 \
#   --frames 120 \
#   --output-html out/min_angle_graph_fast_viewer_video.html \
#   --frame-dir out/min_angle_graph_frames \
#   --video-out out/min_angle_graph_fast_viewer.webm \
#   --workers "$(python3 - <<'PY'
# import os
# print(max(1, (os.cpu_count() or 2) // 2))
# PY
# )"
