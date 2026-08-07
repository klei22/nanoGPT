# Fine-tuning experiments

This directory contains self-contained experiments that fine-tune Hugging Face
models without changing nanoGPT's core training path. Each experiment should
document its hypothesis, data choices, and a reproducible smoke-test command.

## Experiments

* [`rmsnorm_channel_zeroing`](rmsnorm_channel_zeroing/README.md) tests whether
  low-magnitude RMSNorm channels can be driven to zero while retaining language
  modeling performance on a chosen data distribution.
