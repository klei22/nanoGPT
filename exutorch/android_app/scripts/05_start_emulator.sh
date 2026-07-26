#!/usr/bin/env bash
set -euo pipefail

AVD_NAME="${AVD_NAME:-nanogpt_avd}"

if [[ "${HEADLESS:-0}" == "1" ]]; then
  emulator -avd "${AVD_NAME}" -no-window -no-audio
else
  emulator -avd "${AVD_NAME}"
fi
