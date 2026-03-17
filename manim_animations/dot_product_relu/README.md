# Dot Product ReLU Animation

3Blue1Brown-style Manim animations visualizing dot products of vectors with ReLU activation.

## Concept

A random unit vector **w** is dotted with 10 random unit vectors **x_i** on the unit sphere. Only the positive portions (ReLU) are kept and accumulated:

```
sum = Σ ReLU(w · x_i)  for i = 1..10
```

## Scenes

1. **DotProductReLUScene** (3D): Shows the unit sphere, primary vector, and each random vector with its projection. Green = positive dot product, Red = negative (zeroed by ReLU). Running total displayed.

2. **DotProductReLU2D** (2D): Bar chart showing all dot products with ReLU highlighting.

## Usage

```bash
# Install manim
pip install manim

# Render at medium quality
chmod +x render.sh
./render.sh

# Or render individually
manim render -qm dot_product_relu.py DotProductReLUScene
manim render -qm dot_product_relu.py DotProductReLU2D

# Low quality (fast preview)
manim render -ql dot_product_relu.py DotProductReLUScene
```
