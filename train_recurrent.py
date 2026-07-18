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
recur_parser.add_argument("--skip_steps",    type=int, default=0,
                          help="Mask loss for the first K positions in every block")
recur_parser.add_argument("--weight_start",  type=float, default=1.0)
recur_parser.add_argument("--weight_end",    type=float, default=1.0)
recur_parser.add_argument("--reset_optim", action="store_true", help="Ignore optimiser state in the checkpoint")
recur_parser.add_argument("--reset_best_val_loss_on_resume", default=False, action=argparse.BooleanOptionalAction,
                          help="Reset best_val_loss instead of inheriting it from --resume_ckpt")
recur_parser.add_argument("--latent_mix_mode", choices=("direct", "slerp", "add_norm"), default="direct",
                          help="How to turn a recurrent latent into the next input: reuse it directly, slerp toward the correct token embedding, or add then L2-normalize")
recur_parser.add_argument("--latent_mix_alpha", type=float, default=0.5,
                          help="Blend amount for --latent_mix_mode=slerp; ignored by direct/add_norm unless --learn_latent_mix_alpha is set for checkpointing consistency")
recur_parser.add_argument("--learn_latent_mix_alpha", action="store_true",
                          help="Learn the slerp blend amount as a sigmoid-constrained scalar")

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

latent_mix_logit = None
if args.learn_latent_mix_alpha:
    initial_alpha = min(max(args.latent_mix_alpha, 1e-6), 1.0 - 1e-6)
    latent_mix_logit = torch.nn.Parameter(
        torch.logit(torch.tensor(initial_alpha, device=device))
    )
    if ckpt.get("latent_mix_logit") is not None:
        latent_mix_logit.data.copy_(ckpt["latent_mix_logit"].to(device))

decay, no_decay = [], []
for n, p in model.named_parameters():
    (decay if p.dim() >= 2 else no_decay).append(p)
if latent_mix_logit is not None:
    no_decay.append(latent_mix_logit)

param_groups = [
    {"params": decay,     "weight_decay": args.opt_weight_decay},
    {"params": no_decay,  "weight_decay": 0.0},
]

optimizer = optimizer_dictionary[args.optimizer](param_groups, args)

if ckpt.get("optimizer") and not getattr(args, "reset_optim", False):
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
    except ValueError as exc:
        print(f"warning: optimizer state not loaded ({exc})")


best_val_loss_raw = ckpt.get("best_val_loss", float("inf"))
best_val_loss = best_val_loss_raw.item() if hasattr(best_val_loss_raw, "item") else float(best_val_loss_raw)
if args.reset_best_val_loss_on_resume:
    best_val_loss = float("inf")
print("best_val_loss", best_val_loss)
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
# 5)  LATENT MIXING + ONE BLOCK (B,T)  →  scalar loss
# ----------------------------------------------------------------------
def current_latent_mix_alpha():
    if latent_mix_logit is not None:
        return torch.sigmoid(latent_mix_logit)
    return torch.tensor(args.latent_mix_alpha, device=device)


def slerp_latent(latent, target, amount, eps=1e-8):
    """
    Spherical interpolation from `latent` toward the correct-token embedding.
    Norms are interpolated linearly so the result keeps a sensible magnitude
    even when the two input vectors have different lengths.
    """
    latent_norm = latent.norm(dim=-1, keepdim=True).clamp_min(eps)
    target_norm = target.norm(dim=-1, keepdim=True).clamp_min(eps)
    latent_unit = latent / latent_norm
    target_unit = target / target_norm
    dot = (latent_unit * target_unit).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega).clamp_min(eps)
    mixed_unit = (
        torch.sin((1.0 - amount) * omega) / sin_omega * latent_unit
        + torch.sin(amount * omega) / sin_omega * target_unit
    )
    mixed_norm = (1.0 - amount) * latent_norm + amount * target_norm
    return mixed_unit * mixed_norm


def mix_recurrent_latent(latent, correct_token_emb):
    if args.latent_mix_mode == "direct":
        return latent
    if args.latent_mix_mode == "slerp":
        return slerp_latent(latent, correct_token_emb, current_latent_mix_alpha())
    if args.latent_mix_mode == "add_norm":
        mixed = latent + correct_token_emb
        return F.normalize(mixed, p=2, dim=-1) * math.sqrt(gpt_conf.n_embd)
    raise ValueError(f"unknown latent_mix_mode: {args.latent_mix_mode}")


def train_block(x_tokens, y_tokens):
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
        correct_piece = embed_tokens(x_tokens[:, t:t+1])  # GT token embedding
        # ↳ Seed with GT embeddings, then feed back the previous latent.
        #    Optional modes can bias that latent back toward the correct token.
        if t < args.latent_steps or hidden_prev is None:
            new_piece = correct_piece
        else:
            new_piece = mix_recurrent_latent(hidden_prev, correct_piece)

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
            loss = train_block(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            global global_step
            global_step += 1

            if global_step % args.log_interval == 0:
                print(f"iter {global_step:>7} | "
                      f"loss {loss.item():.4f}")
            # ─── validation & checkpoint ───────────────────────────────
            if global_step % args.eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    val = run_epoch("val")          # one full pass

                if tb:
                    tb.add_scalar("loss/val", val, global_step)

                if val < 5.00:
                    print(val)
                    best_val_loss = val
                    torch.save(
                        {"model": model.state_dict(),
                         "model_args": ckpt["model_args"],
                         "iter_num":   global_step,
                         "best_val_loss": best_val_loss,
                         "latent_mix_logit": None if latent_mix_logit is None else latent_mix_logit.detach().cpu()},
                        best_ckpt_path)
                    print(f"  ➜ new best @ step {global_step}; "
                          f"checkpoint saved to {best_ckpt_path}")


        else:

            model.eval()
            with torch.no_grad():
                loss = train_block(x, y)

        losses.append(loss.item())
        ptr += block_size
    return sum(losses) / len(losses)

# ----------------------------------------------------------------------
# 7)  TRAINING DRIVER
# ----------------------------------------------------------------------
tb = SummaryWriter() if getattr(args, "tensorboard_log", False) else None
os.makedirs(args.out_dir, exist_ok=True)
best_ckpt_path = os.path.join(args.out_dir, "ckpt.pt")

val_loss = 999.9
while global_step < args.max_iters:
    t0 = time.time()
    train_loss = run_epoch("train")

    # ── run validation/checkpoint every N iterations ────────────────
    if global_step % args.eval_interval == 0:
        val_loss = run_epoch("val")

        if tb:
            tb.add_scalar("loss/val", val_loss, global_step)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"model": model.state_dict(),
                 "model_args": ckpt["model_args"],
                 "iter_num":   global_step,
                 "best_val_loss": best_val_loss,
                 "latent_mix_logit": None if latent_mix_logit is None else latent_mix_logit.detach().cpu()},
                best_ckpt_path)
            print(f"  ➜ new best @ step {global_step}; "
                  f"checkpoint saved to {best_ckpt_path}")

    # (tensorboard train-loss stays per-epoch to avoid spam)

    if tb:
        tb.add_scalar("loss/train", train_loss, global_step)

    print(f"iter {global_step:03d} | "
          f"train {train_loss:.4f} | val {val_loss:.4f} | "
          f"{(time.time()-t0):.1f}s")


if tb:
    tb.flush()
    tb.close()

print("done.")
# ======================================================================

