# Kernel-accelerated ReLU2Max attention

## What ReLU2Max does

The `relu2max` attention variant replaces softmax with

```text
weight = ReLU(QKᵀ / √d)² / divisor
```

and, when `--div_by_seq_len` is enabled, divides the result by the key sequence
length as well. This is the squared-ReLU attention transformation explored by
[Primer](https://arxiv.org/abs/2109.08668). Unlike softmax, it is not
row-normalized: changing `relu2max_divisor` therefore changes the scale of the
attention output. Causal masking remains correct because masked `-inf` logits
become zero after ReLU.

The existing manual attention path materializes `QKᵀ`, applies the selected
transformation, and multiplies by `V`. The accelerated implementation only
fuses the pointwise ReLU, square, and scaling operations; it does **not** yet
fuse the two matrix multiplications or avoid materializing the quadratic score
matrix. It is consequently a safe incremental optimization, not a ReLU2Max
equivalent of FlashAttention.

## Using the kernel

Run training with:

```bash
python train.py \
  --softmax_variant_attn relu2max \
  --relu2max_use_kernel \
  --relu2max_divisor 256 \
  --div_by_seq_len
```

`--relu2max_use_kernel` sends the pointwise transform through
[`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html).
PyTorch Inductor specializes it lazily for the input device, dtype, and shape;
on CUDA, Inductor normally emits a Triton kernel. AOTAutograd generates the
corresponding backward kernel, so the option works in both training and
inference. The default remains the eager implementation to avoid first-call
compilation cost on short runs and preserve compatibility with older PyTorch
installations.

## Measuring the result

Benchmark after a warm-up call so compilation is excluded. Compare otherwise
identical runs with and without `--relu2max_use_kernel`, and report both step
time and peak memory. The expected benefit is limited to fewer pointwise kernel
launches and score-matrix reads/writes. A future fully fused kernel should tile
`QKᵀ`, apply causal masking and squared ReLU in SRAM, immediately accumulate
against `V`, and implement matching backward kernels; that larger change would
remove the `T × T` score allocation and provide the FlashAttention-like gain.
