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

## Sequential run_experiments demo

`sequential_run_experiments_demo.sh` runs a small sequential pipeline via
`optimization_and_search/run_experiments.py`, exercising resume defaults for
`train.py`, `train_recurrent.py`, and `train_mezo.py` using the accompanying
`sequential_run_experiments_demo.yaml` configuration.
