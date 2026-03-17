# train_multi_token_ntp.py
# ======================================================================
# Multi-token next-token prediction (NTP) training
# ======================================================================
# Instead of predicting just the next token, this script predicts
# multiple tokens in sequence using autoregressive sampling within
# each training step. Each sampled token is appended to the context
# for the next prediction. The loss is the sum of cross-entropy losses
# across all prediction steps.
#
# Usage:
#   python train_multi_token_ntp.py --ntp_tokens 2 --dataset minipile ...
#
# The --ntp_tokens flag controls how many tokens ahead to predict:
#   1 = standard NTP (baseline)
#   2 = predict 2 tokens in sequence
#   3 = predict 3 tokens in sequence
# ======================================================================

import argparse
import sys
import os
import time
import math
import json
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from train_args import parse_args as parse_generic_args
from train_variations.optimizer_variants import optimizer_dictionary
from train_variations.loss_variants import build_loss_function
from model import GPT, GPTConfig


# ----------------------------------------------------------------------
# 1) ARGUMENTS
# ----------------------------------------------------------------------
multi_parser = argparse.ArgumentParser(add_help=False)
multi_parser.add_argument(
    "--ntp_tokens", type=int, default=1,
    help="Number of tokens to predict in sequence per training step (1=standard NTP)"
)
multi_parser.add_argument(
    "--ntp_sample_temperature", type=float, default=1.0,
    help="Temperature for sampling intermediate tokens during multi-token NTP"
)

multi_args, remaining = multi_parser.parse_known_args()

# Parse the standard train args on the leftover argv
sys.argv = [sys.argv[0]] + remaining
generic_args, model_group, training_group, logging_group = parse_generic_args()

# Merge
args = generic_args
for k, v in vars(multi_args).items():
    setattr(args, k, v)

# Match lr_decay_iters to max_iters if requested
if args.lr_decay_match_max_iters:
    args.lr_decay_iters = args.max_iters

# ----------------------------------------------------------------------
# 2) SETUP
# ----------------------------------------------------------------------
device = args.device
if device == "cuda" and not torch.cuda.is_available():
    device = "cpu"
device_type = "cuda" if "cuda" in device else "cpu"

torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ptdtype = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}[args.dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

os.makedirs(args.out_dir, exist_ok=True)

# ----------------------------------------------------------------------
# 3) DATA
# ----------------------------------------------------------------------
data_dir = os.path.join("data", args.dataset)

train_data = np.memmap(
    os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r"
)
val_data = np.memmap(
    os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r"
)


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(
        len(data) - args.block_size - args.ntp_tokens,
        (args.batch_size,),
    )
    # x: input context of length block_size
    x = torch.stack(
        [torch.from_numpy(data[i : i + args.block_size].astype(np.int64)) for i in ix]
    )
    # y: targets — we need block_size + ntp_tokens - 1 tokens of ground truth
    # for multi-step prediction, but we'll index into them step by step
    y = torch.stack(
        [
            torch.from_numpy(
                data[i + 1 : i + 1 + args.block_size + args.ntp_tokens - 1].astype(
                    np.int64
                )
            )
            for i in ix
        ]
    )
    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# ----------------------------------------------------------------------
# 4) MODEL
# ----------------------------------------------------------------------
meta_path = os.path.join(data_dir, "meta.pkl")
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    vocab_size = meta.get("vocab_size", 50304)
else:
    vocab_size = 50304

model_args = {
    action.dest: getattr(args, action.dest) for action in model_group._group_actions
}
model_args["vocab_size"] = vocab_size
model_args["eval_interval"] = args.eval_interval

gptconf = GPTConfig(**model_args)
model = GPT(gptconf).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

if args.compile:
    print("Compiling model...")
    model = torch.compile(model)

# ----------------------------------------------------------------------
# 5) OPTIMIZER
# ----------------------------------------------------------------------
optimizer = model.configure_optimizers(
    args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type
)
scaler = torch.amp.GradScaler(device_type, enabled=(args.dtype == "float16"))


# ----------------------------------------------------------------------
# 6) LEARNING RATE SCHEDULE
# ----------------------------------------------------------------------
def get_lr(it):
    # Linear warmup
    if it < args.warmup_iters:
        return args.learning_rate * it / args.warmup_iters
    # Cosine decay after warmup
    if it > args.lr_decay_iters:
        return args.min_lr
    decay_ratio = (it - args.warmup_iters) / (
        args.lr_decay_iters - args.warmup_iters
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)


# ----------------------------------------------------------------------
# 7) MULTI-TOKEN NTP FORWARD PASS
# ----------------------------------------------------------------------
def multi_token_ntp_step(model, x, y_full, ntp_tokens, temperature, step_seed):
    """
    Perform a multi-token NTP forward pass.

    Args:
        model: The GPT model
        x: Input tensor of shape (B, T)
        y_full: Target tensor of shape (B, T + ntp_tokens - 1)
        ntp_tokens: Number of tokens to predict in sequence
        temperature: Sampling temperature for intermediate tokens
        step_seed: Base seed for reproducible but varying sampling

    Returns:
        total_loss: Sum of cross-entropy losses across all prediction steps
        per_step_losses: List of individual step losses for logging
    """
    total_loss = torch.zeros(1, device=x.device, dtype=torch.float32)
    per_step_losses = []
    current_input = x  # (B, T)

    for k in range(ntp_tokens):
        # Target for this step: the (k)-th token ahead
        # y_full[:, k : k + block_size] gives us the target for each position
        # But we want the target for the LAST position prediction in each step
        # For step k, we predict position T+k given context ending at T+k-1
        targets_k = y_full[:, k : k + args.block_size]  # (B, T)

        # Forward pass
        logits, loss = model(current_input, targets=targets_k)

        if loss is None:
            raise RuntimeError("Model returned None loss — targets may be misaligned")

        total_loss = total_loss + loss
        per_step_losses.append(loss.item())

        # If not the last step, sample the next token and shift input
        if k < ntp_tokens - 1:
            with torch.no_grad():
                # Use a different seed for each step's sampling
                rng = torch.Generator(device=x.device)
                rng.manual_seed(step_seed + k)

                # Get logits for the last position only
                last_logits = logits[:, -1, :] / temperature  # (B, vocab_size)

                # Sample from the distribution
                probs = F.softmax(last_logits, dim=-1)
                sampled_token = torch.multinomial(
                    probs, num_samples=1, generator=rng
                )  # (B, 1)

            # Shift input: drop the first token, append the sampled token
            current_input = torch.cat(
                [current_input[:, 1:], sampled_token], dim=1
            )  # (B, T)

    return total_loss, per_step_losses


# ----------------------------------------------------------------------
# 8) EVALUATION
# ----------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model, ntp_tokens, temperature):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            X, Y = get_batch(split)
            seed = args.seed + k * 1000
            with ctx:
                total_loss, _ = multi_token_ntp_step(
                    model, X, Y, ntp_tokens, temperature, seed
                )
            losses[k] = total_loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ----------------------------------------------------------------------
# 9) TRAINING LOOP
# ----------------------------------------------------------------------
print(f"Multi-token NTP training with ntp_tokens={args.ntp_tokens}")
print(f"Sampling temperature: {args.ntp_sample_temperature}")
print(f"Dataset: {args.dataset}")
print(f"Block size: {args.block_size}, Batch size: {args.batch_size}")
print(f"Max iters: {args.max_iters}")

# Save config
config_out = {
    "ntp_tokens": args.ntp_tokens,
    "ntp_sample_temperature": args.ntp_sample_temperature,
    "dataset": args.dataset,
    "block_size": args.block_size,
    "batch_size": args.batch_size,
    "n_layer": args.n_layer,
    "n_head": args.n_head,
    "n_embd": args.n_embd,
    "max_iters": args.max_iters,
    "learning_rate": args.learning_rate,
}
with open(os.path.join(args.out_dir, "multi_token_ntp_config.json"), "w") as f:
    json.dump(config_out, f, indent=2)

# Tensorboard
writer = None
if args.tensorboard_log:
    tb_name = getattr(args, "tensorboard_run_name", None) or args.out_dir
    writer = SummaryWriter(log_dir=os.path.join("runs", tb_name))

best_val_loss = float("inf")
best_iter = 0
best_tokens = 0
tokens_trained = 0
iter_num = 0
t0 = time.time()

model.train()
X, Y = get_batch("train")

while iter_num < args.max_iters:
    # Set learning rate
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # Evaluation
    if iter_num % args.eval_interval == 0:
        losses = estimate_loss(model, args.ntp_tokens, args.ntp_sample_temperature)
        train_loss = losses["train"]
        val_loss = losses["val"]

        print(
            f"iter {iter_num:>6d} | "
            f"train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"lr {lr:.2e} | "
            f"tokens {tokens_trained:,} | "
            f"ntp_tokens {args.ntp_tokens}"
        )

        if writer:
            writer.add_scalar("Loss/train", train_loss, iter_num)
            writer.add_scalar("Loss/val", val_loss, iter_num)
            writer.add_scalar("lr", lr, iter_num)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_iter = iter_num
            best_tokens = tokens_trained

            if not getattr(args, "never_save_checkpoint", False):
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config_out,
                }
                torch.save(ckpt, os.path.join(args.out_dir, "ckpt.pt"))

        # Write metrics file (compatible with exploration runner)
        num_params = sum(p.numel() for p in model.parameters())
        with open(
            os.path.join(args.out_dir, "best_val_loss_and_iter.txt"), "w"
        ) as f:
            f.write(f"{best_val_loss:.6f}, {best_iter}, {best_tokens}, {num_params}")

    if getattr(args, "eval_only", False):
        break

    # Training step with gradient accumulation
    for micro_step in range(args.gradient_accumulation_steps):
        # Different seed for each iteration + micro_step for sampling diversity
        step_seed = args.seed + iter_num * 100 + micro_step * 7

        with ctx:
            total_loss, per_step_losses = multi_token_ntp_step(
                model, X, Y, args.ntp_tokens, args.ntp_sample_temperature, step_seed
            )
            loss = total_loss / args.gradient_accumulation_steps

        scaler.scale(loss).backward()

        tokens_trained += args.batch_size * args.block_size
        X, Y = get_batch("train")

    if args.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Timing
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % args.log_interval == 0:
        lossf = total_loss.item()
        step_losses_str = " + ".join(f"{l:.4f}" for l in per_step_losses)
        print(
            f"  step {iter_num:>6d} | loss {lossf:.4f} [{step_losses_str}] | "
            f"dt {dt*1000:.0f}ms | lr {lr:.2e}"
        )

    iter_num += 1

# Final summary
print(f"\nTraining complete.")
print(f"  ntp_tokens: {args.ntp_tokens}")
print(f"  best_val_loss: {best_val_loss:.6f}")
print(f"  best_iter: {best_iter}")
print(f"  best_tokens: {best_tokens:,}")

if writer:
    writer.close()
