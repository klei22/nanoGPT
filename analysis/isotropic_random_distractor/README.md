# Isotropic Random-Distractor Model demos

This directory contains standard-library Python demos for the model in the prompt.
They simulate spherical random distractor logits, compare them with the fixed-vocabulary
expansion, and render an HTML report with instructions for reading each figure.

## Run

```bash
python analysis/isotropic_random_distractor/scripts/run_isotropic_demo.py \
  --config analysis/isotropic_random_distractor/configs/base_demo.yaml \
  --outdir analysis/isotropic_random_distractor/reports/base_demo
```

Open `analysis/isotropic_random_distractor/reports/base_demo/index.html` in a browser.

## Configs

- `configs/base_demo.yaml`: balanced baseline.
- `configs/stress_mean_field.yaml`: small vocabulary and high noise to expose the Jensen gap.
- `configs/large_vocab.yaml`: larger vocabulary where the compact mean-field closure is closer.

## Report facets

The generated page covers dimension scaling, finite-vocabulary versus mean-field behavior,
margin regimes, added logit-noise/quantization, and conditional architecture/compute exponents.
CSV files are emitted alongside the HTML for further analysis.
