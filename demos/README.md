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
2. Grouped asymmetric vector PTQ (`--granularity vector --quantization asymmetric`
   with either `--vector-group-count` or `--vector-group-size`).

Example matching 300-d embeddings split into 10 groups (30 values/group):

```bash
bash demos/fake_ptq_asymmetric_grouped_vector_eval_demo_shakespeare_char.sh \
  --vector-group-count 10
```
