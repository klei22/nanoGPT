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

## Snap-to-Grid Projections

`snap_to_grid_demo.sh` prepares the Shakespeare character dataset, trains a
small model with snap-to-grid enabled, evaluates multiple grid sizes, and then
generates text with and without the projections. Run it from the repository
root:

```bash
bash demos/snap_to_grid_demo.sh
```
