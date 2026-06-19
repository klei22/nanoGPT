# Recurrent Fine-Tuning

This guide covers the `train_recurrent.py` script used for latent-chaining fine-tuning.

## Overview

`train_recurrent.py` continues training from a pre-existing checkpoint while feeding the model's hidden state back as the next token for a number of steps. This allows the model to refine its latent representations without using the embedding layer for every position.

## Basic Usage

Assuming you have an existing checkpoint in `out/ckpt.pt`, run:

```bash
python3 train_recurrent.py --resume_ckpt out/ckpt.pt --latent_steps 4 --max_iters 1000
```

The script now supports mixed teacher forcing via `--latent_schedule` as well as
an `--auto_latent` mode that grows the chain length whenever validation loss
improves. TensorBoard will report perplexity and gradient norm statistics.

Important flags include:

- `--latent_steps`: number of hidden states to chain before teacher forcing.
- `--latent_schedule`: comma-separated list of latent step counts to mix each iteration.
- `--auto_latent`: automatically increase latent steps when validation improves.
- `--latent_max_steps`: upper bound when using `--auto_latent`.
- `--skip_steps`: ignore loss on the first positions of every block.
- `--reset_best_val_loss`: begin saving checkpoints immediately regardless of the previous best value.

## Suggested Extensions

- Mixed teacher-forcing schedules instead of a fixed `--latent_steps` value.
- Support for variable-length recurrent chains driven by the validation loss.
- Better statistics reporting such as perplexity or gradient norms during training.
- Integration with streaming data loaders to keep memory usage low.
- Exploration of hierarchical latent chaining across multiple blocks.
- Option to disable teacher forcing entirely once validation loss stabilises.

