#!/usr/bin/env bash
set -euo pipefail

sdkmanager --licenses

sdkmanager \
  "platform-tools" \
  "platforms;android-34" \
  "build-tools;34.0.0" \
  "emulator" \
  "system-images;android-34;google_apis;x86_64"
