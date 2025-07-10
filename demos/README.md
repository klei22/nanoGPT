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

## SpinQuant PTQ

`spinquant_ptq.py` loads a checkpoint produced by `train.py`, learns SpinQuant
rotations on a small calibration set and writes a new quantized checkpoint.

```bash
python3 demos/spinquant_ptq.py --in_dir out --out_dir spinquant_out
```

After running the PTQ demo you can perform inference from the produced checkpoint:

```bash
python3 demos/spinquant_sample.py --ckpt spinquant_out/ckpt_spinquant.pt --prompt "Hello" --max_new_tokens 40
```
