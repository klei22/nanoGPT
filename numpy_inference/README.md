# Numpy Based Inference

This directory contains a minimal numpy implementation of the GPT
inference stack.  It is intended for experimenting with post training
quantization (PTQ) strategies and custom sampling methods directly on
the exported weights from the PyTorch models.

Weights can be exported using `quantization/save_weights.py` which
produces a pickle file of numpy arrays.  The classes here load that
file and perform inference without torch.

The goal is to provide fine grained control of the precision used for
each tensor.  The `quantization` utilities include simple symmetric and
affine schemes that operate purely on numpy arrays.  This makes it
straight forward to emulate int8, int4 or other custom formats such as
bfloat16 or E4M3.

This code is intentionally light weight and is not feature complete
with the PyTorch model.  It mirrors the folder structure of the
`variations` package so new layer implementations can be added easily.

## Example Usage

1. Train a model to produce a checkpoint in an output directory (e.g. `out/ckpt.pt`):

```bash
python3 train.py --out_dir=out
```

2. Export the weights to a numpy pickle using `save_weights.py`:

```bash
python3 quantization/save_weights.py --out_dir=out --file_name=gpt_weights
```

3. Run numpy inference with the exported weights:

```bash
python3 numpy_inference/inference.py --weights gpt_weights.pkl --tokens 20
```

You can experiment with quantization by specifying `--quant_bits`, e.g.:

```bash
python3 numpy_inference/inference.py --weights gpt_weights.pkl --quant_bits 8 --tokens 20
```

By default generation uses `--top_k 40` and `--top_p 0.95`. To change these values:

```bash
python3 numpy_inference/inference.py --weights gpt_weights.pkl \
    --tokens 20 --top_k 50 --top_p 0.9
```
