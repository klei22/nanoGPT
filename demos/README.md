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

## Softmax-1 with OrthoAdam
Run a small demo training using the paper\x27s optimizer and softmax variant:
```bash
bash demos/orthoadam_softmax1_demo.sh
```
This demo also sets `--ortho_seed` for reproducibility.

