#!/usr/bin/env python3
"""Export offline teacher logits for distillation.

This script loads a checkpointed teacher model, runs it over the tokenized
dataset split in fixed-length windows, and stores the logits in a .npy file
with shape [num_starts, block_size, vocab_size].
"""

from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import torch

from gpt_conf import GPTConfig
from model import GPT


def load_meta(dataset: str) -> dict:
    meta_path = os.path.join("data", dataset, "meta.pkl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.pkl not found at {meta_path}")
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def load_tokens(dataset: str, split: str, vocab_size: int) -> np.memmap:
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    dtype = np.uint32 if vocab_size == 100277 else np.uint16
    path = os.path.join("data", dataset, f"{split}.bin")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{split}.bin not found at {path}")
    return np.memmap(path, dtype=dtype, mode="r")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export offline distillation logits.")
    parser.add_argument("--ckpt_path", required=True, help="Path to teacher checkpoint.")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., shakespeare_char).")
    parser.add_argument("--split", default="train", choices=["train", "val"], help="Dataset split to export.")
    parser.add_argument("--output", required=True, help="Output .npy path for logits.")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of windows per batch.")
    parser.add_argument("--device", default="cpu", help="Device for inference (cpu or cuda).")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"], help="Output dtype.")
    parser.add_argument("--block_size", type=int, default=None, help="Override block size for export.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    model_args = checkpoint.get("model_args")
    if model_args is None:
        raise ValueError("Checkpoint is missing model_args.")

    if args.block_size is not None:
        model_args["block_size"] = args.block_size

    meta = load_meta(args.dataset)
    vocab_size = meta.get("vocab_size")
    if vocab_size is None:
        raise ValueError("meta.pkl does not contain vocab_size.")
    model_args["vocab_size"] = vocab_size

    config = GPTConfig(**model_args)
    model = GPT(config)

    state_dict = checkpoint["model"]
    for key in list(state_dict.keys()):
        if key.startswith("_orig_mod."):
            state_dict[key[len("_orig_mod."):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device(args.device)
    model.to(device)

    data = load_tokens(args.dataset, args.split, vocab_size)
    block_size = config.block_size
    num_starts = len(data) - block_size
    if num_starts <= 0:
        raise ValueError("Dataset is shorter than block_size.")

    output_dtype = np.float16 if args.dtype == "float16" else np.float32
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    logits_out = np.lib.format.open_memmap(
        args.output,
        mode="w+",
        dtype=output_dtype,
        shape=(num_starts, block_size, vocab_size),
    )

    with torch.no_grad():
        for start in range(0, num_starts, args.batch_size):
            end = min(start + args.batch_size, num_starts)
            batch_starts = np.arange(start, end)
            batch = np.stack([data[i : i + block_size].astype(np.int64) for i in batch_starts])
            x = torch.from_numpy(batch).to(device)
            logits, _ = model(x, targets=None, iter_num=None, loss_fn=None)
            logits_out[start:end] = logits.detach().cpu().numpy().astype(output_dtype)

    logits_out.flush()
    print(f"Saved logits to {args.output} with shape {logits_out.shape}.")


if __name__ == "__main__":
    main()
