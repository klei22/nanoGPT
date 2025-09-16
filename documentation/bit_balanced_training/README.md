# Bit-Balanced Training Pipeline

This document explains how the adaptive bit-width linear layer communicates its
bit usage to the training loop and how the new bit-balanced loss consumes that
signal alongside the cross-entropy objective.

## 1. Adaptive linear layers report their bit cost

`AdaptiveBitLinear` extends `nn.Linear` with a learnable bit-width parameter
that is stochastically rounded with a straight-through estimator and applied to
both the weights and, optionally, the activations.【F:variations/linear_variations.py†L87-L158】
The layer exposes a `bit_usage()` method that multiplies the rounded bit-width
by the number of stored parameters so downstream utilities can query its
current cost.【F:variations/linear_variations.py†L160-L169】
Any other layer that needs to participate in the pipeline can implement the same
method signature.

## 2. Utilities aggregate the model-wide footprint

`utils.bit_usage.compute_total_bit_usage` walks the module tree, calls each
layer's `bit_usage()` hook, and accumulates the result on the correct
device so it can participate in autograd.【F:utils/bit_usage.py†L19-L44】
If you need per-layer introspection, `collect_bit_usage` returns a dictionary of
each named module's contribution.【F:utils/bit_usage.py†L47-L65】

## 3. The loss adds a configurable bit penalty

`BitBalancedCrossEntropy` wraps the standard cross-entropy objective, queries the
model-wide bit usage via the helper above, and blends both terms using a
configurable weight. Optionally, it normalizes the bit count by the number of
trainable parameters to produce a scale-invariant regularizer.【F:train_variations/loss_variants.py†L33-L72】
The loss object exposes `set_model(model)` so the trainer can provide the module
whose layers implement `bit_usage()`.

## 4. Trainer plumbing provides the model reference

The loss dictionary registers `bit_balanced_cross_entropy` so it is available as
`--loss_fn`. When the trainer builds the requested loss function, it instantiates
`BitBalancedCrossEntropy` with the CLI-provided weight and normalization options.【F:train_variations/loss_variants.py†L342-L511】
After model creation (and DDP wrapping, if enabled) the trainer passes the raw
model to the loss via `set_model`, enabling the penalty term during
training.【F:train.py†L383-L401】

## 5. Command-line controls

Two new CLI flags control the regularizer strength:

* `--bit_loss_weight` sets the multiplier applied to the aggregated bit count.
* `--bit_loss_normalize/--no-bit_loss_normalize` toggles the optional
  parameter-count normalization.

These arguments live in `train_args.py`, alongside the `--loss_fn` selection, so
experiments can sweep over them via `explorations/` YAML definitions or direct
command-line overrides.【F:train_args.py†L87-L149】
