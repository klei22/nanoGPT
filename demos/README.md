# Demos

This folder will hold repeatable demonstrations of features and results.

## Shifted GELU

This shows that the GELU wants to shift:


Call this from either repo root dir, or from demos dir, but ensure the relative
path for ckpt_path is from your present directory.

Example for calling from repo root directory:

```bash
python3 demos/check_ckpt_for_gelu_shift.py \
        --ckpt_path out/ckpt.pt
```

## Optimizer Comparison

`adam_vs_adamw.sh` trains two tiny Shakespeare models, one with Adam and one
with AdamW, then compares their statistics using `view_model_stats.py`.

## Grouped Asymmetric Vector PTQ Comparison

`fake_ptq_asymmetric_grouped_vector_eval_demo_shakespeare_char.sh` runs a
bit-width sweep (default int8 down to int3) on `shakespeare_char` and compares:

1. Original full-vector PTQ (`--granularity vector`, symmetric).
2. Grouped asymmetric vector PTQ (`--granularity vector --quantization asymmetric`)
   while sweeping group-count per vector (default 1 through 10).
3. Grouped symmetric vector PTQ (`--granularity vector --quantization symmetric`)
   over the same group-count sweep.

The demo now gracefully skips requested group-count values that do not evenly
divide the embedding dimension.

Example with default group-count sweep 1..10 for 300-d embeddings:

```bash
bash demos/fake_ptq_asymmetric_grouped_vector_eval_demo_shakespeare_char.sh
```

Example constraining the group-count sweep to 1..10 explicitly:

```bash
bash demos/fake_ptq_asymmetric_grouped_vector_eval_demo_shakespeare_char.sh \
  --group-count-start 1 \
  --group-count-stop 10 \
  --group-count-step 1
```

## Writer-Subspace vs Fake PTQ

`writer_subspace_vs_ptq_demo.sh` trains or reuses a compact `shakespeare_char`
full-precision checkpoint, evaluates it, creates fake PTQ checkpoints across
symmetric/asymmetric schemes and several per-vector group-count settings,
creates writer-subspace checkpoints over the same bit-width sweep, and applies
those fake PTQ settings on top of writer-subspace checkpoints to test the
combined approach. It writes a CSV plus an HTML report with all stats, a
validation-loss-vs-bits chart that uses a dotted horizontal full-precision
reference line, and a validation-loss-vs-quantized-size chart.

```bash
bash demos/writer_subspace_vs_ptq_demo.sh \
  --bit-start 8 \
  --bit-stop 3 \
  --bit-step -1 \
  --writer-attn-rank 128 \
  --writer-mlp-rank 128 \
  --ptq-quantizations "symmetric asymmetric" \
  --ptq-group-counts "0 2 4"
```
