"""Train a sparse autoencoder on hidden activations from a GPT checkpoint.

This script mirrors the standard training workflow by pulling batches from
an on-disk dataset, running them through a frozen GPT model, and
optimizing a sparse autoencoder on the captured activations. The
autoencoder learns a sparse latent representation that can be inspected or
reused for interpretability experiments.
"""
from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn

from model import GPT, GPTConfig
from utils.sparse_autoencoder import SparseAutoencoder, SparseAutoencoderConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a GPT checkpoint (ckpt.pt).")
    parser.add_argument("--dataset", default="shakespeare_char", help="Dataset name under data/ to stream tokens from.")
    parser.add_argument("--block_size", type=int, default=None, help="Context window used for activation capture. Default is the checkpoint block size.")
    parser.add_argument("--batch_size", type=int, default=2, help="Token batches to stream through the GPT model.")
    parser.add_argument("--train_steps", type=int, default=500, help="Number of autoencoder optimization steps to run.")
    parser.add_argument("--eval_interval", type=int, default=50, help="How often to compute validation reconstruction loss.")
    parser.add_argument("--save_interval", type=int, default=200, help="How often to save the autoencoder checkpoint.")
    parser.add_argument("--out_dir", default="out/sparse_autoencoder", help="Directory for autoencoder checkpoints and logs.")
    parser.add_argument("--latent_dim", type=int, default=512, help="Latent width for the autoencoder.")
    parser.add_argument("--l1_alpha", type=float, default=1e-3, help="Weight for the latent L1 sparsity term.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout applied to the latent activations.")
    parser.add_argument("--activation", choices=["relu", "gelu", "silu"], default="relu", help="Non-linearity applied to the latent activations.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Optimizer learning rate for the autoencoder.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay applied to the autoencoder weights.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed used for data shuffling.")
    parser.add_argument(
        "--activation_source",
        choices=["block_output", "residual_pre", "mlp", "attn"],
        default="block_output",
        help="Which part of the transformer block to hook for activations.",
    )
    parser.add_argument("--layer", type=int, default=-1, help="Transformer block index to capture activations from.")
    parser.add_argument(
        "--max_tokens_per_batch",
        type=int,
        default=4096,
        help="Subsample up to this many tokens per batch for autoencoder training to control memory.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Unused placeholder to mirror train.py arguments for downstream tooling.",
    )
    parser.add_argument(
        "--compile_gpt",
        action="store_true",
        help="Compile the frozen GPT model to speed up activation collection (PyTorch 2.x only).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_dataset(dataset: str, block_size: int) -> Tuple[np.memmap, np.memmap, int]:
    data_root = os.path.join("data", dataset)
    meta_path = os.path.join(data_root, "meta.pkl")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Could not find dataset metadata at {meta_path}. Run the dataset's prepare.py first.")

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    vocab_size = meta.get("vocab_size", None)
    if vocab_size is None:
        raise ValueError("meta.pkl is missing a vocab_size entry; cannot infer token dtype.")

    dtype = np.uint32 if vocab_size == 100277 else np.uint16
    train_path = os.path.join(data_root, "train.bin")
    val_path = os.path.join(data_root, "val.bin")
    train_data = np.memmap(train_path, dtype=dtype, mode="r")
    val_data = np.memmap(val_path, dtype=dtype, mode="r")

    if len(train_data) <= block_size or len(val_data) <= block_size:
        raise ValueError(f"Dataset sequences must be longer than the requested block size {block_size}.")

    return train_data, val_data, vocab_size


def get_batch(data: np.memmap, block_size: int, batch_size: int, device: torch.device) -> torch.Tensor:
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
    )
    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
    return x


def instantiate_gpt(checkpoint_path: str, device: torch.device, block_size_override: int | None, compile_gpt: bool) -> GPT:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint["model_args"]
    if block_size_override is not None:
        model_args["block_size"] = block_size_override
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    for k in list(state_dict.keys()):
        if k.startswith("_orig_mod."):
            state_dict[k[len("_orig_mod."):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    if compile_gpt and hasattr(torch, "compile"):
        model = torch.compile(model)

    return model


def resolve_capture_target(model: GPT, layer: int, source: str) -> Tuple[nn.Module, bool]:
    blocks = model.transformer["h"]
    if layer < 0:
        layer = len(blocks) + layer
    if layer < 0 or layer >= len(blocks):
        raise ValueError(f"Layer index {layer} is out of range for {len(blocks)} layers.")

    block = blocks[layer]
    if source == "block_output":
        return block, False
    if source == "residual_pre":
        return block, True
    if source == "mlp":
        return block.mlp, False
    if source == "attn":
        return block.attn, False
    raise ValueError(f"Unknown activation source: {source}")


def capture_activations(
    model: GPT,
    tokens: torch.Tensor,
    capture_module: nn.Module,
    use_prehook: bool,
) -> torch.Tensor:
    buffer: Dict[str, torch.Tensor] = {}

    def hook(_module, inputs, output=None):
        tensor = inputs[0] if use_prehook else output
        buffer["activation"] = tensor.detach()

    handle = (
        capture_module.register_forward_pre_hook(hook)
        if use_prehook
        else capture_module.register_forward_hook(hook)
    )

    with torch.no_grad():
        model(tokens)
    handle.remove()

    if "activation" not in buffer:
        raise RuntimeError("Activation hook did not capture any data.")
    return buffer["activation"]


def subsample_tokens(hidden: torch.Tensor, max_tokens: int) -> torch.Tensor:
    flat = hidden.reshape(-1, hidden.size(-1))
    if max_tokens is not None and flat.size(0) > max_tokens:
        idx = torch.randperm(flat.size(0), device=flat.device)[:max_tokens]
        flat = flat[idx]
    return flat


def evaluate_autoencoder(
    model: GPT,
    autoencoder: SparseAutoencoder,
    val_data: np.memmap,
    capture_module: nn.Module,
    use_prehook: bool,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    autoencoder.eval()
    with torch.no_grad():
        tokens = get_batch(val_data, args.block_size, args.batch_size, device)
        hidden = capture_activations(model, tokens, capture_module, use_prehook)
        hidden = subsample_tokens(hidden, args.max_tokens_per_batch)
        recon, latent = autoencoder(hidden)
        loss = autoencoder.loss(recon, latent, hidden)
    autoencoder.train()
    return float(loss.item())


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    gpt_model = instantiate_gpt(args.checkpoint, device, args.block_size, args.compile_gpt)
    if args.block_size is None:
        args.block_size = gpt_model.config.block_size

    train_data, val_data, _ = load_dataset(args.dataset, args.block_size)

    capture_module, use_prehook = resolve_capture_target(gpt_model, args.layer, args.activation_source)

    ae_config = SparseAutoencoderConfig(
        input_dim=gpt_model.config.n_embd,
        hidden_dim=args.latent_dim,
        dropout=args.dropout,
        activation=args.activation,
        l1_alpha=args.l1_alpha,
    )
    autoencoder = SparseAutoencoder(ae_config).to(device)
    optimizer = torch.optim.AdamW(
        autoencoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    for step in range(1, args.train_steps + 1):
        tokens = get_batch(train_data, args.block_size, args.batch_size, device)
        hidden = capture_activations(gpt_model, tokens, capture_module, use_prehook)
        hidden = subsample_tokens(hidden, args.max_tokens_per_batch)

        optimizer.zero_grad(set_to_none=True)
        recon, latent = autoencoder(hidden)
        loss = autoencoder.loss(recon, latent, hidden)
        loss.backward()
        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.eval_interval == 0 or step == 1:
            val_loss = evaluate_autoencoder(
                gpt_model, autoencoder, val_data, capture_module, use_prehook, args, device
            )
            print(
                f"step {step:04d} | train_loss {loss.item():.4f} | val_loss {val_loss:.4f} | sparsity {latent.abs().mean().item():.4f}"
            )

        if step % args.save_interval == 0 or step == args.train_steps:
            ckpt_path = os.path.join(args.out_dir, f"sparse_autoencoder_step_{step:06d}.pt")
            torch.save(
                {
                    "autoencoder": autoencoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": asdict(ae_config),
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"Saved sparse autoencoder checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
