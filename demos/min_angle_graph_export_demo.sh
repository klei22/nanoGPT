#!/bin/bash
set -euo pipefail

# Demo the per-validation LM-head minimum-angle graph export on the small
# exploration config, then point the user at the local Plotly viewer.
#
# Optional environment variables:
#   OUTPUT_DIR  Base output directory for train.py runs. Default: out
#   PREFIX      Prefix for exploration run names. Default: min_angle_graph_demo_

OUTPUT_DIR="${OUTPUT_DIR:-out}"
PREFIX="${PREFIX:-min_angle_graph_demo_}"
CONFIG="explorations/min_angle_graph_export.yaml"
VIEWER="analysis/min_angle_graph_plotly_viewer.html"

python3 optimization_and_search/run_experiments.py \
  --config "${CONFIG}" \
  --config_format yaml \
  --output_dir "${OUTPUT_DIR}" \
  --prefix "${PREFIX}"

cat <<EOF

Minimum-angle graph export demo completed.

CSV/JSON exports are written under:
  out/min_angle_graph_exports/<run-name>/

Open the Plotly viewer in your browser:
  ${VIEWER}

Then select one or more exported CSV files from an export directory to step
through validation snapshots from first iteration to last.
EOF
