#!/usr/bin/env bash
# Generate Conway-like cellular automata CSV data and convert it to per-column
# integer multicontext datasets.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_ROOT="conway_life_mc_int"
INPUT_CSV="${SCRIPT_DIR}/input.csv"
WIDTH=8
HEIGHT=8
EPISODES=4
STEPS=16
SEED=1337
ALIVE_VALUE=255
MUTATION_CHANCE=0.015
TRAIN_RATIO=0.9
SAVE_VALUES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
    --output_csv) INPUT_CSV="$2"; shift 2 ;;
    --width) WIDTH="$2"; shift 2 ;;
    --height) HEIGHT="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --alive_value) ALIVE_VALUE="$2"; shift 2 ;;
    --mutation_chance) MUTATION_CHANCE="$2"; shift 2 ;;
    --train_ratio) TRAIN_RATIO="$2"; shift 2 ;;
    --save_values_csv) SAVE_VALUES=1; shift ;;
    -h|--help)
      cat <<EOF
Usage: $0 [options]

Options:
  --output_root NAME       Dataset root under data/ (default: conway_life_mc_int)
  --output_csv PATH        CSV path to generate (default: data/conway_life_mc_int/input.csv)
  --width N                Grid width (default: 8)
  --height N               Grid height (default: 8)
  --episodes N             Independent rollouts (default: 4)
  --steps N                Frames per rollout (default: 16)
  --seed N                 RNG seed (default: 1337)
  --alive_value 1|255      Pixel value for live cells (default: 255)
  --mutation_chance P      Per-cell flip chance per step (default: 0.015)
  --train_ratio P          Train split ratio (default: 0.9)
  --save_values_csv        Also write values.csv inside every column dataset
EOF
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

python3 "${SCRIPT_DIR}/generate_conway_life_csv.py" \
  --output_csv "${INPUT_CSV}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --episodes "${EPISODES}" \
  --steps "${STEPS}" \
  --seed "${SEED}" \
  --alive_value "${ALIVE_VALUE}" \
  --mutation_chance "${MUTATION_CHANCE}"

PIXEL_MAX=$((ALIVE_VALUE))
if [[ "${PIXEL_MAX}" -lt 1 ]]; then
  PIXEL_MAX=1
fi
CELL_COUNT=$((WIDTH * HEIGHT))
RANGE_ARGS=(
  --output_root "${OUTPUT_ROOT}"
  --train_ratio "${TRAIN_RATIO}"
  --range "timestamp:0:999"
  --range "episode:0:$((EPISODES - 1))"
  --range "step:0:$((STEPS - 1))"
  --range "width:${WIDTH}:${WIDTH}"
  --range "height:${HEIGHT}:${HEIGHT}"
  --range "rule_id:0:3"
  --range "pattern_id:0:6"
  --range "density_percent:0:100"
  --range "mutation_per_mille:0:1000"
  --range "alive_count:0:${CELL_COUNT}"
  --range "born_count:0:${CELL_COUNT}"
  --range "died_count:0:${CELL_COUNT}"
)

for ((i = 0; i < CELL_COUNT; i++)); do
  RANGE_ARGS+=(--range "p${i}:0:${PIXEL_MAX}")
done

if [[ "${SAVE_VALUES}" -eq 1 ]]; then
  RANGE_ARGS+=(--save_values_csv)
fi

"${REPO_ROOT}/data/csv_mc_int/get_dataset.sh" "${INPUT_CSV}" "${RANGE_ARGS[@]}"
