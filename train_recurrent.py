# ======================================================================
# train_recurrent.py  –  latent-chaining fine-tuning
# ======================================================================
#  * resumes from an existing checkpoint (no scratch / no GPT-2 import)
#  * feeds the HIDDEN state (after ln_f / scale_down) back as the next
#    “token”, skipping de-embedding, for the first `--latent_steps`
#  * keeps cross-entropy vs. ground-truth, with optional per-position
#    linear weighting and an initial “skip” window
# ----------------------------------------------------------------------

import argparse
import sys
import os
import time
import math
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from train_variations.optimizer_variants import optimizer_dictionary
from train_args import parse_args as parse_generic_args

from model import GPT, GPTConfig           # your patched model.py


global_step = 0          # counts *training* iterations


# ----------------------------------------------------------------------
# 1)  ARGUMENTS  –  reuse *everything* from train_args.py
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# 1-bis)  add the *extra* flags that are unique to latent-chaining
# ----------------------------------------------------------------------
recur_parser = argparse.ArgumentParser(add_help=False)
recur_parser.add_argument("--resume_ckpt",   required=True,
                          help="Path to .pt checkpoint produced by train.py")
recur_parser.add_argument("--latent_steps",  type=int, default=0,
                          help="Chain this many hidden states before teacher-forcing")
recur_parser.add_argument("--latent_schedule", type=str, default="",
                          help="Comma-separated schedule of latent steps; overrides --latent_steps")
recur_parser.add_argument("--auto_latent", action="store_true",
                          help="Increase latent steps on validation improvement")
recur_parser.add_argument("--latent_max_steps", type=int, default=0,
                          help="Maximum latent steps when using --auto_latent")
recur_parser.add_argument("--skip_steps",    type=int, default=0,
                          help="Mask loss for the first K positions in every block")
recur_parser.add_argument("--weight_start",  type=float, default=1.0)
recur_parser.add_argument("--weight_end",    type=float, default=1.0)
recur_parser.add_argument("--reset_optim", action="store_true", help="Ignore optimiser state in the checkpoint")
recur_parser.add_argument("--reset_best_val_loss", action="store_true",
                          help="Start saving recurrent checkpoints from the beginning")

# -- split cmdline -----------------------------------------------------
latent_args, remaining = recur_parser.parse_known_args()

# ----------------------------------------------------------------------
# 1-b)  now run the gigantic parser **only on the leftovers**
# ----------------------------------------------------------------------
sys.argv = [sys.argv[0]] + remaining          # fake argv for train_args
generic_args, *_ = parse_generic_args()

# ----------------------------------------------------------------------
# 1-c)  merge both Namespaces into one `args`
# ----------------------------------------------------------------------
args = generic_args
for k, v in vars(latent_args).items():
    setattr(args, k, v)

if args.latent_schedule:
    args.latent_schedule = [int(x) for x in args.latent_schedule.split(',')]
else:
    args.latent_schedule = [args.latent_steps]
if args.latent_max_steps == 0:
    args.latent_max_steps = max(args.latent_schedule)
curr_latent_steps = args.latent_schedule[0]

def choose_latent_steps():
    """Return latent steps for this iteration."""
    if args.auto_latent:
        return curr_latent_steps
    return random.choice(args.latent_schedule)


# ----------------------------------------------------------------------
# 2)  LOAD CHECKPOINT + MODEL
# ----------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt   = torch.load(args.resume_ckpt, map_location=device)

gpt_conf = GPTConfig(**ckpt["model_args"])
model    = GPT(gpt_conf).to(device)

def unwrap_state_dict(wrapped_sd):
    """
    Remove '_orig_mod.' (torch.compile) and 'module.' (DDP) prefixes so the
    keys match a plain, single-GPU GPT instance.
    """
    clean = {}
    for k, v in wrapped_sd.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        if k.startswith("module."):
            k = k[len("module."):]
        clean[k] = v
    return clean

state_dict = unwrap_state_dict(ckpt["model"])
missing, unexpected = model.load_state_dict(state_dict, strict=False)

if missing:
    print(f"warning: {len(missing)} missing params (OK if all zero-grad)")
if unexpected:
    print(f"warning: {len(unexpected)} extra params ignored")

# helpers exposed in patched model.py
embed_tokens     = model.embed_tokens
forward_embedded = lambda x: model.forward_embedded(x, return_hidden=True)

decay, no_decay = [], []
for n, p in model.named_parameters():
    (decay if p.dim() >= 2 else no_decay).append(p)

param_groups = [
    {"params": decay,     "weight_decay": args.opt_weight_decay},
    {"params": no_decay,  "weight_decay": 0.0},
]

optimizer = optimizer_dictionary[args.optimizer](param_groups, args)

if ckpt.get("optimizer") and not getattr(args, "reset_optim", False):
    optimizer.load_state_dict(ckpt["optimizer"])


best_val_loss = ckpt["best_val_loss"].item()
print("best_val_loss", best_val_loss)
if args.reset_best_val_loss:
    best_val_loss = 5.0
iter_num      = ckpt["iter_num"]          # not used, but preserved

block_size = gpt_conf.block_size

# ----------------------------------------------------------------------
# 3)  DATA (mmap – same layout as train.py)
# ----------------------------------------------------------------------
def load_bin(split):
    path = os.path.join("data", args.dataset, f"{split}.bin")
    return np.memmap(path, dtype=np.uint16, mode="r")

train_bin, val_bin = load_bin("train"), load_bin("val")

# ----------------------------------------------------------------------
# 4)  LOSS-WEIGHT HELPER
# ----------------------------------------------------------------------
def make_loss_weights(bsz: int, T: int, device):
    w = torch.linspace(args.weight_start, args.weight_end, steps=T,
                       device=device).repeat(bsz, 1)
    if args.skip_steps:
        w[:, :args.skip_steps] = 0.0
    return w

# ----------------------------------------------------------------------
# 5)  ONE BLOCK (B,T)  →  scalar loss
# ----------------------------------------------------------------------
def train_block(x_tokens, y_tokens, latent_steps):
    """
    One recurrent block that **preserves full self-attention context**.
    We build a `hidden_buf` (B, ≤T, E); at each step we append either
    the latent vector from the previous step or the ground-truth embedding,
    then run the whole sequence through the model once.
    """
    B, T   = x_tokens.shape
    device = x_tokens.device

    weights = make_loss_weights(B, T, device)
    nz_sum  = weights.sum() + 1e-8

    hidden_buf  = None        # grows (B,t,E)
    hidden_prev = None        # last latent state (B,1,E)
    total_loss  = 0.0

    for t in range(T):
        # ---- decide what to append ---------------------------------
        # ↳ Teacher-force **until** we reach latent_steps
        if t < latent_steps:
            new_piece = embed_tokens(x_tokens[:, t:t+1])  # GT token
        else:
            new_piece = hidden_prev                       # latent

        # ---- grow the buffer ---------------------------------------
        hidden_buf = new_piece if hidden_buf is None else \
                     torch.cat([hidden_buf, new_piece], dim=1)

        # ---- full forward pass on the whole buffer -----------------
        logits_all, h_all = forward_embedded(hidden_buf)

        logits_step = logits_all[:, -1, :]   # newest position only
        hidden_prev = h_all[:,  -1:, :]      # keep for next iteration

        ce = F.cross_entropy(logits_step, y_tokens[:, t], reduction="none")
        total_loss += (ce * weights[:, t]).sum()

    return total_loss / nz_sum

# ----------------------------------------------------------------------
# 6)  EPOCH LOOPS
# ----------------------------------------------------------------------
def run_epoch(split):
    global best_val_loss, curr_latent_steps
    data   = train_bin if split == "train" else val_bin
    losses = []
    ptr    = 0
    while ptr + block_size + 1 < len(data):
        seq = torch.from_numpy(np.array(
                  data[ptr:ptr + block_size + 1], dtype=np.int64)
              ).to(device)
        x, y = seq[:-1].unsqueeze(0), seq[1:].unsqueeze(0)  # (1,T)

        if split == "train":
            model.train()
            optimizer.zero_grad()
            loss = train_block(x, y, choose_latent_steps())
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            global global_step
            global_step += 1

            if global_step % args.log_interval == 0:
                ppl = math.exp(loss.item())
                print(f"iter {global_step:>7} | "
                      f"loss {loss.item():.4f} | ppl {ppl:.2f} | grad {grad_norm:.2f}")
                if tb:
                    tb.add_scalar("grad_norm", grad_norm, global_step)
                    tb.add_scalar("ppl/train", ppl, global_step)
            # ─── validation & checkpoint ───────────────────────────────
            if global_step % args.eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    val = run_epoch("val")          # one full pass

                if tb:
                    tb.add_scalar("loss/val", val, global_step)
                    tb.add_scalar("ppl/val", math.exp(val), global_step)

                if val < best_val_loss:
                    print(val)
                    best_val_loss = val
                    if args.auto_latent and curr_latent_steps < args.latent_max_steps:
                        curr_latent_steps += 1
                        print(f"latent_steps -> {curr_latent_steps}")
                    torch.save(
                        {"model": model.state_dict(),
                         "model_args": ckpt["model_args"],
                         "iter_num":   global_step,
                         "best_val_loss": best_val_loss},
                        best_ckpt_path)
                    print(f"  ➜ new best @ step {global_step}; "
                          f"checkpoint saved to {best_ckpt_path}")


        else:

            model.eval()
            with torch.no_grad():
                loss = train_block(x, y, choose_latent_steps())

        losses.append(loss.item())
        ptr += block_size
    return sum(losses) / len(losses)

# ----------------------------------------------------------------------
# 7)  TRAINING DRIVER
# ----------------------------------------------------------------------
tb = SummaryWriter() if getattr(args, "tensorboard_log", False) else None
best_ckpt_path = os.path.join(os.path.dirname(args.resume_ckpt), "ckpt_lat.pt")

val_loss = 999.9
while global_step < args.max_iters:
    t0 = time.time()
    train_loss = run_epoch("train")

    # ── run validation/checkpoint every N iterations ────────────────
    if global_step % args.eval_interval == 0:
        val_loss = run_epoch("val")

        if tb:
            tb.add_scalar("loss/val", val_loss, global_step)
            tb.add_scalar("ppl/val_epoch", math.exp(val_loss), global_step)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if args.auto_latent and curr_latent_steps < args.latent_max_steps:
                curr_latent_steps += 1
                print(f"latent_steps -> {curr_latent_steps}")
            torch.save(
                {"model": model.state_dict(),
                 "model_args": ckpt["model_args"],
                 "iter_num":   global_step,
                 "best_val_loss": best_val_loss},
                best_ckpt_path)
            print(f"  ➜ new best @ step {global_step}; "
                  f"checkpoint saved to {best_ckpt_path}")

    # (tensorboard train-loss stays per-epoch to avoid spam)

    if tb:
        tb.add_scalar("loss/train", train_loss, global_step)
        tb.add_scalar("ppl/train_epoch", math.exp(train_loss), global_step)

    ppl_train = math.exp(train_loss)
    ppl_val   = math.exp(val_loss)
    print(f"iter {global_step:03d} | "
          f"train {train_loss:.4f} (ppl {ppl_train:.2f}) | "
          f"val {val_loss:.4f} (ppl {ppl_val:.2f}) | "
          f"{(time.time()-t0):.1f}s")


if tb:
    tb.flush()
    tb.close()

print("done.")
# ======================================================================

