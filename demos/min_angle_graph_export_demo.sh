#!/bin/bash

# Demo the per-validation LM-head minimum-angle graph export on the small
# exploration config, then point the user at the local Plotly viewer.
#
# Optional environment variables:
#   OUTPUT_DIR  Base output directory for train.py runs. Default: out
#   PREFIX      Prefix for exploration run names. Default: timestamped demo prefix
#   EXPORT_ROOT Directory where the exploration config writes CSV/JSON exports.
#               Default: out/min_angle_graph_exports

OUTPUT_DIR="${OUTPUT_DIR:-out}"
PREFIX="${PREFIX:-min_angle_graph_demo_$(date +%Y%m%d_%H%M%S)_}"
EXPORT_ROOT="${EXPORT_ROOT:-out/min_angle_graph_exports}"
CONFIG="explorations/min_angle_graph_export.yaml"
VIEWER="analysis/min_angle_graph_plotly_viewer.html"
FAST_VIEWER="analysis/min_angle_graph_fast_viewer.html"
PREPROCESSOR="utils/preprocess_min_angle_graph_csvs.py"

before_csv_count="$(find "${EXPORT_ROOT}" -type f -name "${PREFIX}*.csv" 2>/dev/null | wc -l | tr -d ' ')"

cat <<EOF
Starting LM-head minimum-angle graph export demo.

Config:      ${CONFIG}
Output dir:  ${OUTPUT_DIR}
Export root: ${EXPORT_ROOT}
Run prefix:  ${PREFIX}

This runs the shakespeare_char and minipile smoke-test configurations. The
minipile export scans a much larger vocabulary, so it can take noticeably
longer than the shakespeare_char run even though the training run is tiny.
EOF

python3 optimization_and_search/run_experiments.py \
  --config "${CONFIG}" \
  --config_format yaml \
  --output_dir "${OUTPUT_DIR}" \
  --prefix "${PREFIX}"

after_csv_count="$(find "${EXPORT_ROOT}" -type f -name "${PREFIX}*.csv" 2>/dev/null | wc -l | tr -d ' ')"
if [[ "${after_csv_count}" -le "${before_csv_count}" ]]; then
  cat >&2 <<EOF

Minimum-angle graph export demo did not produce new CSV exports.

The experiment runner may have reported a failed train.py run without exiting
non-zero. Review the run output above and fix the training error, then rerun
this demo. Expected new CSV files matching:
  ${EXPORT_ROOT}/${PREFIX}*.csv
EOF
  exit 1
fi

cat <<EOF

Minimum-angle graph export demo completed.

CSV/JSON exports are written under:
  ${EXPORT_ROOT}/<run-name>/

New CSV exports detected:
$(find "${EXPORT_ROOT}" -type f -name "${PREFIX}*.csv" 2>/dev/null | sort)

For small vocabularies, open the original Plotly viewer in your browser:
  ${VIEWER}

For large vocabularies or many snapshots, preprocess the unchanged CSV exports
and open the fast viewer instead:
  python3 ${PREPROCESSOR} "${EXPORT_ROOT}/<run-name>" --output_dir "${EXPORT_ROOT}/<run-name>/fast_viewer"
  ${FAST_VIEWER}

Then select the generated *.min-angle-graph.json files in the fast viewer to
step through validation snapshots from first iteration to last with bounded
WebGL point counts and precomputed histograms.
EOF
