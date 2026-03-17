#!/bin/bash
# Render the dot product ReLU manim animations
#
# Usage:
#   ./render.sh              # Render both scenes at medium quality
#   ./render.sh -ql          # Low quality (fast preview)
#   ./render.sh -qh          # High quality
#   ./render.sh -qk          # 4K quality
#
# Scenes:
#   DotProductReLUScene  - 3D animation with unit sphere
#   DotProductReLU2D     - 2D bar chart companion

QUALITY="${1:--qm}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Rendering 3D Dot Product ReLU Scene ==="
manim render "$QUALITY" "$SCRIPT_DIR/dot_product_relu.py" DotProductReLUScene

echo ""
echo "=== Rendering 2D Bar Chart Scene ==="
manim render "$QUALITY" "$SCRIPT_DIR/dot_product_relu.py" DotProductReLU2D

echo ""
echo "=== Done! ==="
echo "Output files are in: $SCRIPT_DIR/media/"
