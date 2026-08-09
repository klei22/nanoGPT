#!/usr/bin/env bash
set -euo pipefail

AVD_NAME="${AVD_NAME:-nanogpt_avd}"
AVD_PACKAGE="${AVD_PACKAGE:-system-images;android-34;google_apis;x86_64}"
AVD_DEVICE="${AVD_DEVICE:-pixel}"

avdmanager create avd \
  --name "${AVD_NAME}" \
  --package "${AVD_PACKAGE}" \
  --device "${AVD_DEVICE}"
