# Isotropic Random-Distractor Model Demo

This directory contains a reproducible Monte Carlo/analytic probe for the note in the user request.
It tests the model's facets without training a language model:

- exact spherical moment identities (`E[u^2]=1/d`, `E[u^4]=3/(d(d+2))`),
- the fixed-vocabulary softmax expansion,
- the Gaussian/mean-field upper-curve diagnostic and partition concentration,
- margin-dependent context sensitivity,
- anisotropy via an effective dimension,
- structured hard-negative tails as an assumption violation,
- added zero-mean logit noise,
- conditional parameter and compute scaling algebra.

Run:

```bash
bash demos/isotropic_random_distractor_demo.sh
```

For a faster smoke test:

```bash
bash demos/isotropic_random_distractor_demo.sh \
  explorations/isotropic_random_distractor_fast.yaml \
  report/isotropic_random_distractor_fast
```

The output report is an HTML file with Plotly graphs. Solid lines/points are Monte Carlo estimates; dashed lines are analytic predictions.

## Minipile trained-model sweep

The toy Monte Carlo report is complemented by a trained-model workflow for `minipile`:

```bash
bash demos/isotropic_random_distractor_minipile_sweep.sh
```

That wrapper:

1. prepares the `data/minipile` tokenized dataset if needed,
2. runs `optimization_and_search/run_from_yaml.py` on `explorations/isotropic_random_distractor_minipile_train.yaml`,
3. keeps checkpoints so trained unembedding geometry can be measured, and
4. calls `analysis/isotropic_random_distractor/analyze_minipile_sweep.py` to write a Plotly report under `report/isotropic_random_distractor_minipile/`.

The minipile analyzer reads the training YAML log, fits validation-loss trends against `1 / n_embd` and parameter count, and optionally inspects checkpoint readout geometry (`lm_head.weight` or tied `transformer.wte.weight`) for pairwise cosine variance and participation-ratio effective dimension. These checks are empirical diagnostics of the assumptions in the note, not a claim that isotropic distractors explain all minipile loss.
