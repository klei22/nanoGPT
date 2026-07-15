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
