# Recurrent Fine-Tuning

This guide covers the `train_recurrent.py` script used for latent-chaining fine-tuning.

## Overview

`train_recurrent.py` continues training from a pre-existing checkpoint while feeding the model's hidden state back as the next token for a number of steps. This allows the model to refine its latent representations without using the embedding layer for every position.

## Basic Usage

Assuming you have an existing checkpoint in `out/ckpt.pt`, run:

```bash
python3 train_recurrent.py --resume_ckpt out/ckpt.pt --latent_steps 4 --max_iters 1000
```

Important flags include:

- `--latent_steps`: number of hidden states to chain before teacher forcing.
- `--skip_steps`: ignore loss on the first positions of every block.
- `--reset_best_val_loss`: begin saving checkpoints immediately regardless of the previous best value.

## Suggested Extensions

- Mixed teacher-forcing schedules instead of a fixed `--latent_steps` value.
- Support for variable-length recurrent chains driven by the validation loss.
- Better statistics reporting such as perplexity or gradient norms during training.

