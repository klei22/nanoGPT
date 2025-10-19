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

## Snap-to-Grid Evaluation

`snap_to_grid_demo.sh` fine-tunes a compact Shakespeare model while sweeping
different snap-to-grid sizes, persists the generated registries, evaluates
validation loss for each configuration, and finally samples text with every
grid. The script accepts `SNAP_SIZES`, `SNAP_LAYERS`, and `SNAP_COMPONENTS`
environment overrides for quick experimentation.
