#!/usr/bin/env bash
set -euo pipefail

sudo apt update
sudo apt install -y \
  unzip \
  zip \
  curl \
  wget \
  git \
  openjdk-17-jdk \
  libgl1 \
  libpulse0 \
  libxkbcommon0 \
  libxcomposite1 \
  libxdamage1 \
  libxrandr2 \
  libgtk-3-0
