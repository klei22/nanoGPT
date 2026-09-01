# Closing the ReLU2Max–Flash Attention performance gap

The current Triton path is an **elementwise** optimization, not a fused
attention implementation. Attention still performs and stores the complete
`[batch, heads, query, key]` score matrix, launches a separate causal-mask
operation, launches ReLU2Max, and then reads the matrix again for `weights @ V`.
At sequence length 1,024 this quadratic global-memory traffic dominates the
few arithmetic operations saved by fusing ReLU, square, and division.

SDPA/Flash Attention is faster because it tiles Q, K, and V through SRAM,
combines score calculation, masking, normalization, and value accumulation,
and never writes the full score/probability matrix to HBM. Optimizing the
elementwise ReLU2Max kernel further cannot close that architectural gap.

## Proposed FlashReLU2 algorithm

ReLU2Max is unusually convenient for a Flash-style kernel because it has no
row-wise denominator. For query tile `Q_i` and successive key/value tiles
`K_j`, `V_j`, the forward kernel can directly accumulate

```text
S_ij = qk_scale * Q_i @ K_j.T
W_ij = relu(causal_mask(S_ij))**2 / divisor
O_i += W_ij @ V_j
```

in FP32 registers, casting only the final output. No online maximum or running
normalization denominator is required. The score tiles die in SRAM after each
`W_ij @ V_j`, reducing score storage from `O(T²)` to a fixed tile.

Training needs custom recompute-in-backward kernels. Given `D = dL/dO`, each
tile recomputes `S` and `W`, then uses

```text
dV_j += W_ij.T @ D_i
dS_ij = (2 * relu(S_ij) / divisor) * (D_i @ V_j.T)
dQ_i += qk_scale * dS_ij @ K_j
dK_j += qk_scale * dS_ij.T @ Q_i
```

Separate query-owned `dQ` and key-owned `dK/dV` kernels avoid global atomics.
This recomputation trades inexpensive matrix arithmetic for eliminating the
quadratic saved-attention tensor, as Flash Attention does.

## Implementation stages

1. Add a causal forward-only Triton kernel and validate FP32/FP16/BF16 output
   against the materialized implementation across irregular sequence lengths.
2. Add tiled `dQ` and `dK/dV` kernels and use PyTorch `gradcheck` in FP64 on a
   reference implementation plus tolerance tests on CUDA types.
3. Autotune query/key block sizes, head-dimension blocks, warps, and pipeline
   stages for A100. Benchmark head dimensions 64 and 128 and lengths from 128
   through 4,096 rather than tuning only the 1,024-token case.
4. Dispatch only supported causal, contiguous CUDA cases to FlashReLU2. Keep
   the existing elementwise Triton and PyTorch implementations as correctness
   fallbacks for unusual masks, devices, and dtypes.
5. After attention fusion is stable, fuse Q/K L2 normalization and partial
   RoPE into the Q/K load path. This is secondary; eliminating the quadratic
   intermediate is the principal opportunity.

## Measurement requirements

Compare complete forward **and backward** iterations with identical Q/K/V,
dtype, scale, causal semantics, and loss. Report latency and peak allocated
memory. Include generic softmax, SDPA, materialized PyTorch ReLU2Max,
elementwise-Triton ReLU2Max, and fused FlashReLU2 as separate rows. A forward-
only result is useful for kernel development but must not be presented as a
training-speed result.

The realistic target is to approach SDPA by adopting its IO strategy—not by
expecting a faster pointwise activation to compensate for materializing and
moving the `T × T` attention matrix.
