#!/bin/bash
set -e  # Exit on error

# First pass: generate .aux and .bbl
tectonic --keep-intermediates iclr2025.tex

# Run bibtex
bibtex iclr2025

# Re-compile to incorporate bibliography
tectonic --keep-intermediates iclr2025.tex
tectonic iclr2025.tex

echo "✅ Build complete: iclr2025.pdf"

