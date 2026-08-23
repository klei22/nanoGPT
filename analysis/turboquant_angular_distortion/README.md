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
  --no-transformed-tq \
  --tq-bits 3 4 \
  --output outputs/turboquant_isotropic_vs_int.pdf
```

For isotropic Gaussian pairs, a shared random orthogonal transform does not
change the input distribution. Consequently, the transformed and untransformed
TurboQuant curves should agree up to Monte Carlo noise. The sparse mode is
included specifically to reveal why TurboQuant performs the transform before
scalar quantization rather than treating the Lloyd-Max codebook as a standalone
number format. Use `--no-transformed-tq` for isotropic plots to omit those
statistically redundant Hadamard curves.

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

## Angular-space evenness and dispersion

`angle_space_evenness.py` complements the angle-pair distortion curves by
measuring how uniformly each scalar codebook's 3D directions fill the target
sphere. It compares INT3--INT8 and TQ3--TQ8 by default:

```bash
python analysis/turboquant_angular_distortion/angle_space_evenness.py
```

The analysis writes a machine-readable CSV and a six-panel comparison plot. It
samples the same number of code triples for every format, projects them to the
unit sphere, and reports both code-frequency-weighted (`codes`) and deduplicated
angular-support (`unique`) rows. The latter asks how evenly the distinct sampled
directions are placed, while the former also captures directional multiplicity.
The metrics are:

- **HEALPix coverage**: fraction of equal-area target cells reached.
- **Normalized Shannon entropy**: occupancy entropy divided by the entropy of a
  uniform HEALPix map. Higher is more even.
- **Entropy-effective coverage**: `exp(entropy) / number_of_pixels`, the fraction
  of equally populated cells that would have the observed entropy.
- **Jensen-Shannon divergence from uniform**: bounded distributional mismatch;
  zero is ideal.
- **HEALPix count coefficient of variation**: relative cell-count dispersion;
  zero is ideal.
- **Random spherical-cap discrepancy**: the largest sampled difference between
  observed mass in a cap and its uniform area. This detects broad holes or
  clusters that a single grid alignment can miss.
- **Dipole norm and second-moment anisotropy**: rotation-independent tests of
  global directional bias. These remain useful when HEALPix resolution changes.

For a resolution study, run several `--nside` values. Coverage is resolution and
sample-count dependent, so comparisons are only meaningful when those settings
are held fixed. Entropy, divergence, and cap discrepancy measure the
**frequency-weighted representation** obtained by choosing scalar codes
uniformly in the `codes` rows. The `unique` rows instead deduplicate sampled
normalized directions. For large codebooks this is an estimate of unique
support rather than exhaustive enumeration; increase `--samples` when comparing
high widths.

```bash
python analysis/turboquant_angular_distortion/angle_space_evenness.py \
  --formats int3 int4 int8 tq3 tq4 tq8 \
  --samples 500000 --nside 64 --caps 1024 \
  --csv outputs/evenness_nside64.csv \
  --output outputs/evenness_nside64.pdf
```
