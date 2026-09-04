#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MESH_SCRIPT="$PROJECT_ROOT/recurrent_attnres_mesh.py"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

run_stamp() {
  date -u +%Y%m%dT%H%M%SZ
}

require_accelerate() {
  if ! "$PYTHON_BIN" -c 'import accelerate' >/dev/null 2>&1; then
    echo "Accelerate was not found. Run scripts/setup_env.sh first." >&2
    exit 1
  fi
}

require_single_a100_process() {
  local world_size="${WORLD_SIZE:-1}"
  if [[ ! "$world_size" =~ ^[0-9]+$ ]] || [[ "$world_size" -ne 1 ]]; then
    echo "These launchers require WORLD_SIZE=1 (one process on one A100)." >&2
    exit 1
  fi
  if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
    export CUDA_VISIBLE_DEVICES=0
  elif [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
    echo "Set CUDA_VISIBLE_DEVICES to exactly one GPU, not '$CUDA_VISIBLE_DEVICES'." >&2
    exit 1
  fi
}
