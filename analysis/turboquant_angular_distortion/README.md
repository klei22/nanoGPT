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

INT curves use a sequential Viridis palette, while TQ curves use a categorical
red/orange/purple/pink/brown/cyan palette. The categorical TQ colors are
deliberately separated in hue so the dotted codebook-only curves in the
isotropic PDF can be matched to their legend entries at a glance.

## Isotropic dimension sweep

`isotropic_dimension_sweep.py` extends the isotropic comparison across every
power-of-two dimension from 2 through 1024. Each trial generates one isotropic
pair that is shared by every INT/TQ quantizer, reducing comparison noise. The
default uses 100 trials per angle; plots show mean trends with standard-error
bands or bars rather than individual noisy trials.

```bash
bash analysis/turboquant_angular_distortion/demo_isotropic_dimension_sweep.sh
```

The sweep writes five artifacts:

- `isotropic_dimension_curves.csv`: mean and standard deviation at every
  dimension, angle, format, and bit width.
- `isotropic_dimension_summary.csv`: mean absolute, RMS, maximum absolute, and
  signed-bias distortion aggregated over input angles.
- `isotropic_dimension_curves.pdf`: six bit-width panels containing INT and TQ
  mean angle curves, colored by dimension, with translucent mean standard-error
  bands.
- `isotropic_dimension_summary.pdf`: four dimension-scaling panels for the
  aggregate metrics; the mean-absolute-distortion panel includes standard-error
  bars across Monte Carlo trials.
- `isotropic_dimension_tq_advantage.pdf`: INT MAE minus TQ MAE, where positive
  values indicate lower TurboQuant distortion.

Override the range or sampling cost through environment variables:

```bash
MIN_DIM=2 MAX_DIM=256 TRIALS=10 ANGLE_STEP=10 \
  bash analysis/turboquant_angular_distortion/demo_isotropic_dimension_sweep.sh \
  /tmp/isotropic-dimension-quick
```

The direct Python CLI additionally accepts `--bits`, `--angles-start`,
`--angles-stop`, `--clip-sigma`, and `--seed`.

### Separate high-dimensional graph sets

The original HEALPix evenness analysis is specifically three-dimensional: it
projects scalar-code triples onto S2. For dimensions 1024, 2048, 4096, and 8192,
`high_dim_angle_space_evenness.py` replaces HEALPix occupancy with metrics that
remain meaningful on a high-dimensional sphere:

- discrepancy between the observed pair-cosine CDF and the exact uniform-sphere
  beta-distribution CDF;
- observed cosine standard deviation minus the uniform target `1/sqrt(d)`;
- resultant/dipole norm; and
- an exchangeable second-moment anisotropy estimate.

Run the high-dimensional evenness metrics and separate isotropic distortion
graphs for all four dimensions with:

```bash
bash analysis/turboquant_angular_distortion/demo_high_dimensional.sh
```

This writes one evenness CSV/PDF and one isotropic distortion PDF per dimension.
Sampling is streamed in batches so dimension 8192 does not require retaining the
full experiment in memory. A quick run can be requested with:

```bash
EVENNESS_SAMPLES=1000 EVENNESS_BATCH_SIZE=64 \
DISTORTION_TRIALS=5 ANGLE_STEP=15 \
  bash analysis/turboquant_angular_distortion/demo_high_dimensional.sh \
  /tmp/turboquant-high-dimensional
```

The high-dimensional suite is intentionally separate from the default demo
because its four large-dimension Monte Carlo runs are substantially more
expensive.

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
Within every panel, formats are ordered by descending bit width, with the four
direct comparisons adjacent: INT code-frequency, INT unique-support, TQ
code-frequency, and TQ unique-support. Subtle vertical dividers separate bit
widths. The metrics are:

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
