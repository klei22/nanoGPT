#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ANDROID_EXPORT_DIR="${ANDROID_EXPORT_DIR:-${APP_DIR}/../android_export}"

cp "${ANDROID_EXPORT_DIR}/nanogpt_xnnpack.pte" "${APP_DIR}/app/src/main/assets/"
cp "${ANDROID_EXPORT_DIR}/manifest.json" "${APP_DIR}/app/src/main/assets/"

if [[ ! -f "${APP_DIR}/app/src/main/assets/prompt_tokens.txt" ]]; then
  cat > "${APP_DIR}/app/src/main/assets/prompt_tokens.txt" <<'EOF'
464, 3290, 318, 257, 1332
EOF
fi
