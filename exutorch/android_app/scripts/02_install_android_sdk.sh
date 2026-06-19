#!/usr/bin/env bash
set -euo pipefail

ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}"

mkdir -p "${ANDROID_SDK_ROOT}/cmdline-tools"
cd /tmp
curl -fLO https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip
unzip -q commandlinetools-linux-11076708_latest.zip
rm -rf "${ANDROID_SDK_ROOT}/cmdline-tools/latest"
mv cmdline-tools "${ANDROID_SDK_ROOT}/cmdline-tools/latest"

PROFILE_FILE="${PROFILE_FILE:-$HOME/.bashrc}"
{
  echo ""
  echo "# Android SDK"
  echo "export ANDROID_SDK_ROOT=\"${ANDROID_SDK_ROOT}\""
  echo "export PATH=\"${ANDROID_SDK_ROOT}/cmdline-tools/latest/bin:${ANDROID_SDK_ROOT}/platform-tools:${ANDROID_SDK_ROOT}/emulator:\$PATH\""
} >> "${PROFILE_FILE}"

echo "Updated ${PROFILE_FILE} with ANDROID_SDK_ROOT and PATH."
