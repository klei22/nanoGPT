# Vector Distribution Analysis

![e3m2_closeup](screenshot_closeup_e3m2.png)

This tool explores how different number formats affect density of angles created
by sampled vectors after projected onto the unit sphere.

---

## Requirements
- Python 3.x
- numpy, matplotlib, plotly, healpy

Install dependencies (if needed):
```sh
pip install numpy matplotlib plotly healpy
```

## Script Usage: `vector_distribution_analysis.py`

```sh
python vector_distribution_analysis.py --format <format> [--mode exhaustive|random|gaussian] [--num N] [--bins B] [--out PATH] [--healpix] [--nside N] [--out3d PATH] [--exp E] [--mant M] [--mean MU] [--std SIGMA] [--projection equal-area|angular] [--log]
```

### Key Arguments
- `--format`         : Number format. One of `int3` through `int8`, `e4m3`, `e5m2`, `fp16`, `phi`, or `turboquant1` through `turboquant4`.
- `--mode`           : Vector generation mode. `exhaustive` (all possible), `random` (random samples), or `gaussian` (samples from a Gaussian, then quantized). Default: `exhaustive`.
- `--num`            : Number of samples (required for `random` and `gaussian` modes).
- `--mean`           : Mean for Gaussian mode (default: 0.0).
- `--std`            : Standard deviation for Gaussian mode (default: 0.02).
- `--bins`           : Number of bins per dimension for 2D heatmap (default: 60).
- `--out`            : Output path for 2D heatmap image (default: `images/heatmap.png`).
- `--out3d`          : Output path for 3D heatmap (HTML for interactive Plotly, or image for static 3D plot).
- `--healpix`        : Use HEALPix projection for 3D binning and visualization (recommended for 3D output).
- `--nside`          : HEALPix resolution (power of 2, default: 16; higher = finer mesh).
- `-e`, `--exp`      : Number of exponent bits for floating formats (used with `fp16`, `e4m3`, `e5m2`).
- `-m`, `--mant`     : Number of mantissa bits for floating formats.
- `--projection`     : 2D heatmap projection: `equal-area` (default) or `angular`.
- `--log`            : Use log scale for heatmap color.

### Example: 2D and 3D Heatmap

```sh
python3 vector_distribution_analysis.py --format int5 --mode exhaustive --healpix --nside 256 --out images/int5_heatmap.png --out3d images/int5_heatmap3d.html
```

### Example: Gaussian Sampling

```sh
python3 vector_distribution_analysis.py --format fp16 -e 3 -m 2 --mode gaussian --num 100000 --healpix --nside 128 --out3d images/fp16_e3m2_gaussian.html
```

### TurboQuant comparison

TurboQuant is not an IEEE-style scalar storage format. Its MSE variant first
randomly rotates a vector, then represents each rotated coordinate with one of
`2^b` Lloyd-Max centroids optimized for the coordinate distribution. The
`turboquant1` through `turboquant4` options expose the cross-checked
standard-normal centroid tables. Their common dimension-dependent scale is
omitted because it cannot change a vector's direction on the sphere.

Generate the complete comparison suite with:

```sh
bash demo_turboquant.sh
```

Set `NSIDE` to change the HEALPix resolution or pass an output directory:

```sh
NSIDE=64 bash demo_turboquant.sh /tmp/turboquant-surfaces
```

In addition to the six interactive HEALPix surfaces, the demo writes sparse and
isotropic angular-distortion PDFs plus the angular-space evenness CSV/PDF for
INT3--INT8 and TQ3--TQ8. The isotropic plot omits the redundant Hadamard curves,
because isotropic inputs are already rotation invariant. Runtime and resolution
can be adjusted without editing the script:

```sh
NSIDE=16 \
ANGULAR_DIM=1024 ANGULAR_TRIALS=10 ANGLE_STEP=5 \
EVENNESS_SAMPLES=20000 EVENNESS_CAPS=64 \
DIM_SWEEP_TRIALS=10 DIM_SWEEP_ANGLE_STEP=10 \
bash demo_turboquant.sh /tmp/turboquant-quick
```

The default generated analysis files are:

- `turboquant_vs_int_angular_distortion_sparse.pdf`
- `turboquant_vs_int_angular_distortion_isotropic.pdf`
- `angle_space_evenness.csv`
- `angle_space_evenness.pdf`
- `isotropic_dimension_sweep/`, containing curve/summary CSVs and three PDFs
  for dimensions 2 through 1024.

The visualizations enumerate scalar-codebook triples before normalization. They
therefore show the directional geometry of one fixed rotated basis, rather than
averaging over random rotation matrices. This makes the codebook directly
comparable to the integer and floating-point plots.

Implementation references used for the format definition:

- [TurboQuant paper (arXiv:2504.19874)](https://arxiv.org/abs/2504.19874),
  especially Algorithm 1 and the continuous Lloyd-Max objective in Equation 3.
- [Open-source TurboQuant implementation](https://github.com/vivekvar-dl/turboquant),
  which computes dimension-specific Beta-distribution codebooks.
- [Independent reproduction and published-codebook cross-check](https://github.com/ray-ruisun/TurboQuant),
  which records the standard-normal centroid tables used here.

### Output
- 2D heatmaps are saved as images (e.g., PNG) showing vector density on the sphere.
- 3D/HEALPix outputs are interactive HTML files (if `.html` extension is used) or static images, visualizing density on the sphere's surface.

Images generated by this script should be placed in the `images/` directory.

## Batch Automation: `demo.sh`

The `demo.sh` script automates a series of analyses for various formats and modes, generating a suite of 3D HEALPix-based visualizations.

### What it does:
- Runs exhaustive analysis for integer formats (`int3` to `int7`), saving interactive 3D HTML plots.
- Runs exhaustive analysis for custom floating-point formats with varying exponent/mantissa bits.
- Runs Gaussian sampling for selected formats (e.g., `int4`, `fp16` with custom bits), producing large-sample visualizations.

### Demo Script Usage and Reference

```sh
bash demo.sh
```

This will generate a set of `.html` files in the `images/` directory, such as:
- `healpix_int3_exhaustive.html`, `healpix_int4_exhaustive.html`, ...
- `healpix_e3m2_gaussian_100000_healpix_500.html`, etc.

Each file visualizes the density of quantized vectors on the unit sphere for the specified format and sampling mode.


## Requirements

- Python 3.x
- numpy, matplotlib, plotly, healpy

Install dependencies (if needed):
```sh
pip install numpy matplotlib plotly healpy
```

---

## Description

### Vector Projection Onto Unit-Sphere and Binning

To do this, we first sample vectors (if number format small enough we can do one
vector for each possible xyz value), and project it onto the surface of the unit
sphere, and add "1" for that region.

As latitude and longitude lines bunch at the north and south poles, creating
distortion, we suggest utilizing a spherical tessellation as given by HEALPix,
which is very effective at removing distortion artifacts.

### Findings

These show both distinct concentrations for floating point vs integer format,
and different patterns that emerge as mantissa and exponent are increased. There
are also certain clear pathways that emerge that can affect the transition of
weight vectors as each vector is updated via backpropagation or reinforcement
learning.

### Research Next Steps

These insights are critical to study of proper or novel number systems for LLMs,
known to encode information via unique vector directions in hyperdimensional
space.

## Notes
- For best results, ensure the `images/` directory exists before running scripts.
- The 3D HTML outputs are interactive and can be viewed in any modern web browser.
- For custom floating-point formats, use `--exp` and `--mant` to specify exponent and mantissa bits.
