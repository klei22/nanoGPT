#!/usr/bin/env bash
set -euo pipefail

adb devices

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${APP_DIR}"
./gradlew :app:installDebug

adb logcat | grep NanoGPT
