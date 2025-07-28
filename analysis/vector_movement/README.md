# Vector Movement

This tool simulates the training of a 3‑D vector so that it aligns with a target vector.
The vectors are initialized from Gaussian distributions by default but may also be
fixed to one of the coordinate axes or the positive cube corner.

Both the moving vector and the target are fake quantized to integers at each
step.  The bit width of the quantization can be chosen between 3 and 8 bits and
defaults to int4.

A small training loop in PyTorch is run and the movement of the vector is visualised
with Plotly as an interactive HTML animation.

```
python vector_movement.py --help
```
