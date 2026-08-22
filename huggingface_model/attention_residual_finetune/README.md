# Fine-tune the final attention residual

This experiment ports the **last** mixture from Full Attention Residuals to a
pretrained Hugging Face causal language model. It leaves every pretrained
parameter frozen, collects the embedding/layer hidden states, RMS-normalizes
them as keys, applies a token-local softmax over model depth, and sends the
weighted sum through the existing language-model head. Only one pseudo-query
(`residual.query`) is optimized. This changes depth routing, not causal
token-to-token attention.

The zero initialization intentionally matches nanoGPT's attention-residual
implementation: the first forward pass is an equal average over all depths.
It therefore does **not** initially reproduce the base model's logits.

## Run

Install `torch`, `transformers`, `datasets`, and `accelerate`, then run from the
repository root:

```bash
python -m huggingface_model.attention_residual_finetune.train \
  --model gpt2 --max-steps 500
```

The command prints the trainable parameter names before training and saves the
small mixer state dict separately as `final_attention_residual.pt`. Loading the
original base model plus this state dict is sufficient; no frozen model weights
are duplicated.

This first experiment targets decoder-only causal LMs whose output head accepts
hidden states directly. Architectures that require an additional final norm or
an output-head bias should get an architecture-specific adapter before use.
