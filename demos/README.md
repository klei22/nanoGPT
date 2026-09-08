# Demos

This folder will hold repeatable demonstrations of features and results.

## ConvRot fake-W4A4 PTQ

`convrot_ptq_demo.sh` trains a tiny character-level nanoGPT model and runs an
educational, kernel-free adaptation of ConvRot. It applies the paper's
group-wise regular Hadamard transform to both synthetic activations and a
checkpoint weight, verifies that the full-precision linear operation is
unchanged, and compares naive versus rotated per-vector fake W4A4 error.

```bash
bash demos/convrot_ptq_demo.sh
```

The JSON report is written to `out_convrot_ptq_demo/convrot_report.json`.
This demonstrates the numerical PTQ technique only: values remain floating
point, so it does not claim the memory or Tensor Core speedups of a fused INT4
ConvLinear4bit kernel. `GROUP_SIZE` may be any power of four (at least four).

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
