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

`isotropic_dimension_sweep.py` extends the isotropic comparison across the
power-of-two dimensions 256, 512, 1024, and 2048 by default. Each trial generates one isotropic
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
MIN_DIM=256 MAX_DIM=1024 TRIALS=10 ANGLE_STEP=10 \
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
when invoked directly, and is also included in the complete
`vector_distribution/demo_turboquant.sh` suite. Its four large-dimension Monte
Carlo runs are substantially more expensive, so the complete demo exposes
separate `HIGH_DIM_*` environment controls.

## Three- and four-bit all-finite E/M comparison

`low_bit_em_comparison.py` adds naive floating-point baselines without NaN or
infinity encodings. Total width includes one sign bit, so every legal split is:

- 3-bit: E1M1 and E2M0;
- 4-bit: E1M2, E2M1, and E3M0.

Every exponent bit pattern represents a finite number. Exponent zero provides
zero/subnormal values and all other exponent patterns are normal finite values.
Run the new comparison without replacing any existing graphs:

```bash
bash analysis/turboquant_angular_distortion/demo_low_bit_em.sh
```

The output contains curve and summary CSVs, a two-panel 3/4-bit angular
distortion PDF at dimension 2048, and a two-panel dimension-scaling PDF over
256, 512, 1024, and 2048. Each bit-level panel uses a consistent color, line
texture, and marker for INT, TQ, and each E/M split. Mean curves include
standard-error bands and dimension trends include standard-error bars.

## Grouped symmetric/asymmetric and power-of-two scaling sweep

`grouped_quantization_sweep.py` fixes the vector dimension at 2048 and sweeps
contiguous group sizes 16, 32, 64, 128, 256, 512, 1024, and 2048. It compares
four INT4 quantizers:

- symmetric with an exact scalar scale;
- asymmetric with an exact scalar scale and learned integer zero point;
- symmetric with the scale rounded upward to a power of two; and
- asymmetric with a power-of-two scale and integer zero point.

It also plots native-format reference points for:

- **NVFP4**, modeled as E2M1 data, one E4M3FN scale per 16 values, and one FP32
  tensor scale; and
- **MXINT8**, modeled as INT8 data with one E8M0/power-of-two shared scale per
  native 32-value block.

NVFP4 and MXINT8 appear at their native block sizes rather than as artificial
group-size sweeps. The power-of-two INT4 curve is a useful MX-style ablation but
is not labeled MXINT4 because MXINT4 is not an OCP MX v1.0 concrete format.

```bash
bash analysis/turboquant_angular_distortion/demo_grouped_quantization.sh
```

The suite writes angle-level and summary CSVs, a four-panel angular-distortion
PDF colored by group size, and a group-size summary PDF comparing angular MAE
and normalized reconstruction MSE with standard-error bars. Definitions were
cross-checked against the
[NVIDIA Transformer Engine NVFP4 guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html#nvfp4-format)
and the [OCP MX reference implementation](https://github.com/microsoft/microxcaling),
which specifies native MX blocks of 32 and provides MXINT8 as the concrete
integer format.

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
