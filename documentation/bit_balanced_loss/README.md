# Bit-Balanced Adaptive Linear Pipeline

This document explains how learnable bit-width linear layers propagate their
bit-usage metadata into the training loss so that the optimizer can balance
accuracy against model compression.

## 1. AdaptiveBitLinear layers emit differentiable bit counts

The [`AdaptiveBitLinear`](../../variations/linear_variations.py) projection wraps
`nn.Linear` and introduces a learnable `bit_param` that is constrained between
a configurable minimum and maximum number of bits. During the forward pass the
layer fake-quantizes weights (and optionally activations) with a straight
through estimator, while caching the clipped, rounded bit-width. The layer
exposes a `bit_usage()` method that returns `bit_param * parameter_count`,
allowing gradients to flow back into the bit parameter whenever the downstream
loss depends on the reported value. All standard linear configuration flags
(e.g. `--adaptive_linear_init_bits`, `--adaptive_linear_min_bits`, …) are
surfaced through `GPTConfig` and the CLI so the layer can be swapped in via
`--linear_variant_* adaptive_bit_linear`.

## 2. Utility helpers aggregate module-level usage

`utils/bit_usage.py` provides discovery and aggregation helpers that traverse a
model and collect `bit_usage()` outputs from any module that implements the
method. `compute_total_bit_usage(model)` returns the sum of all reported bit
counts as a tensor on the model's device so that it can participate in
backpropagation. The sibling helper `collect_bit_usage(model)` retains the
per-module breakdown for logging or diagnostics.

## 3. The bit-aware loss consumes the aggregated signal

`BitBalancedCrossEntropy` in
[`train_variations/loss_variants.py`](../../train_variations/loss_variants.py)
wraps standard cross entropy and adds a weighted penalty term based on
`compute_total_bit_usage`. The penalty weight (`--bit_loss_weight`) and optional
normalization by the number of trainable parameters (`--bit_loss_normalize`)
are configurable. The loss instance receives a reference to the underlying
model through `set_model(model)`, which the trainer calls after building or
wrapping the network so that DDP shims are transparent.

When invoked during training the loss computes
```
loss = cross_entropy(logits, targets) + bit_loss_weight * total_bits(model)
```
where `total_bits(model)` stays connected to every adaptive layer's
`bit_param`.

## 4. Wiring inside the trainer

`Trainer` constructs the loss through `build_loss_function(args)`, which
returns either a direct callable or a scheduler. The builder instantiates the
bit-balanced loss with the CLI-configured hyperparameters and the trainer calls
`set_model(raw_model)` right after optional compilation or DDP wrapping.
Scheduled losses forward the `set_model` call to any underlying component that
supports it, ensuring the bit usage term is available regardless of the loss
composition.

## 5. Using the loss in experiments

1. Select the adaptive projection, e.g. `--linear_variant_mlp adaptive_bit_linear`
   (or the more specific per-projection flags) and optionally adjust the
   bit-width bounds.
2. Enable the loss with `--loss_fn bit_balanced_cross_entropy` and choose a
   penalty weight such as `--bit_loss_weight 1e-5`.
3. When TensorBoard logging is enabled (`--tensorboard_log`) the trainer will
   emit aggregate metrics under each dataset namespace, including the current
   total bit budget and the contribution of the bit-penalty term to the overall
   loss. For more granular inspection you can still call
   `collect_bit_usage(model)` during evaluation or logging.

The exploration template
[`explorations/bit_balanced_vs_cross_entropy.yaml`](../../explorations/bit_balanced_vs_cross_entropy.yaml)
contrasts the new objective with the standard cross entropy loss to help tune
regularization strengths.
