# Kernel-accelerated ReLU2Max attention

ReLU2Max replaces softmax attention weights with
`relu(QK^T / sqrt(head_dim))^2 / divisor`. The eager path is simple, but writes
the full `T x T` score and weight tensors to memory. That quadratic traffic is
the same bottleneck avoided by scaled-dot-product/Flash Attention.

The fused path uses PyTorch FlexAttention to generate one CUDA attention kernel
containing the score transform and causal mask. It therefore retains the
linear-memory, tiled QK/PV execution model of SDPA while preserving ReLU2Max's
mathematics and autograd support. It is enabled by default for `relu2max` on a
supported PyTorch/CUDA build:

```bash
python train.py --softmax_variant_attn relu2max --use_fused_relu2max
```

Use `--no-use_fused_relu2max` for A/B comparisons. The implementation falls
back automatically on CPU and when attention dropout, FIRE, sliding windows,
logit soft-capping, or learned QK scaling require semantics the kernel does not
currently cover. Quantized attention continues through the existing eager path.

Benchmark both paths at the intended dtype and sequence length after warm-up;
the first fused call includes kernel compilation. Compare peak CUDA memory,
tokens/second, forward/backward time, loss, and gradient norms. The expected
benefit grows with sequence length: intermediate attention storage changes from
quadratic to linear, while arithmetic remains quadratic as it does for SDPA.
