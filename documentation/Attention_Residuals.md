# Attention Residuals

Set `attention_residual_variant: full` (or pass
`--attention_residual_variant full`) to replace the running residual sum with
token-local attention over depth.

For each Transformer block, the implementation:

1. mixes the embedding and earlier sublayer outputs for the attention input;
2. appends the raw self-attention output to depth memory;
3. computes a separate mixture for the MLP input; and
4. appends the raw MLP output.

One final mixture is passed to `ln_f`. Each destination owns a learned
pseudo-query. With the default softmax, queries are initialized to zero, so
every mixture initially is an equal-weight average. Keys are parameter-free RMS-normalized source vectors,
values are the raw vectors, and softmax is only over depth. Consequently this
feature does not replace causal token-to-token self-attention.

The depth weighting defaults to `softmax`. Set
`attention_residual_weighting: relu2max` (or pass
`--attention_residual_weighting relu2max`) to use the repository's ReLU2Max
implementation instead. Its scale follows `relu2max_divisor` and
`div_by_seq_len`, just as it does for token-to-token attention. ReLU2Max routing
queries use a small random initialization instead of zero initialization to
avoid ReLU2's zero-gradient point.

ReLU2Max query initialization can be explored with
`attention_residual_relu2max_query_init`. Supported styles are `zeros`, `ones`,
`constant`, `normal`, `uniform`, `positive_uniform`, `xavier_normal`, and
`xavier_uniform`. Set `attention_residual_relu2max_query_init_scale` to control
the constant value, distribution width, or Xavier gain. In particular, the
combination `ones` and a scale of `1.0` initializes every query component to
one. `zeros` is provided as an ablation but will start ReLU2Max at its
zero-gradient point.

The comparison exploration disables TensorBoard logging so the architecture
sweep does not require optional TensorBoard/TensorFlow binary dependencies.
Remove `tensorboard_log: [false]` from the YAML if TensorBoard is desired and
compatible with the local NumPy environment.

Full Attention Residuals store the embedding and all `2 * n_layer` sublayer
outputs, and perform quadratic work in the number of sublayers. The current
implementation supports sequential attention-then-MLP blocks without post-LN;
the usual PreNorm configuration is supported. Use `standard` (the default) for
the existing additive residual architecture and checkpoint compatibility.
