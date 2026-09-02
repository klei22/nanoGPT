# TurboQuant angular distortion comparison

This analysis places INT3 through INT8 and TQ3 through TQ8 Lloyd-Max codebooks
on the **same angular-distortion graph**. Each TurboQuant width is shown twice:

1. randomized Walsh-Hadamard transform followed by the Gaussian Lloyd-Max
   codebook, modeling TurboQuant's high-dimensional rotation step; and
2. the identical scalar codebook with no transform.

The inverse transform is unnecessary for this metric: an orthogonal inverse
preserves the angle between the two dequantized vectors.

```bash
python analysis/turboquant_angular_distortion/turboquant_angular_distortion.py
```

The default plot uses a sparse pair stress test, where the transform's spreading
effect is visible. To reproduce the random-unit-vector model used by the broader
angular-distortion analysis, run:

```bash
python analysis/turboquant_angular_distortion/turboquant_angular_distortion.py \
  --pair-mode isotropic \
  --tq-bits 3 4 \
  --output outputs/turboquant_isotropic_vs_int.pdf
```

For isotropic Gaussian pairs, a shared random orthogonal transform does not
change the input distribution. Consequently, the transformed and untransformed
TurboQuant curves should agree up to Monte Carlo noise. The sparse mode is
included specifically to reveal why TurboQuant performs the transform before
scalar quantization rather than treating the Lloyd-Max codebook as a standalone
number format.

The implementation uses Gaussian Lloyd-Max codebooks calculated for any
requested `--tq-bits` value rather than copying a short table. The centroids are
scaled by `1/sqrt(dim)`, matching the high-dimensional coordinate model from
[TurboQuant](https://arxiv.org/abs/2504.19874). A randomized Hadamard transform
is used instead of a dense Haar matrix because it is orthogonal and practical at
the default dimension of 4096.

Useful options:

- `--int-bits 3 4 5 6 7 8`: integer curves to overlay.
- `--tq-bits 3 4 5 6 7 8`: one or more TurboQuant codebooks.
- `--pair-mode sparse|isotropic`: structured stress test or random unit pairs.
- `--dim 4096`: power-of-two transform dimension.
- `--trials 30`: Monte Carlo trials per angle and curve.
- `--angles-step 3`: angular sampling interval in degrees.
