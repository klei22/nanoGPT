# Vector Movement

This tool simulates the training of a 3‑D vector so that it aligns with a target vector.
The vectors are initialized from Gaussian distributions by default but may also be
fixed to one of the coordinate axes or the positive cube corner.

Both the moving vector and the target are fake quantized to integers at each
step.  The bit width of the quantization can be chosen between 3 and 8 bits and
defaults to int4.

A small training loop in PyTorch is run and the movement of the vector is visualised
with Plotly as an interactive HTML animation.

The HTML shows two 3‑D views (a plain point plot and a HEALPix sphere) along with
XY, XZ and YZ projections. The XYZ coordinates of the current vector, its
normalized counterpart and the target are printed in the lower-right panel.

You can directly specify the coordinates of either vector using
`--init-x`, `--init-y`, `--init-z` and `--target-x`, `--target-y`, `--target-z`.

```
python vector_movement.py --help
```
