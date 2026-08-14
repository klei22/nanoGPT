# ======================================================================
# train_recurrent.py  –  latent-chaining fine-tuning
# ======================================================================
#  * trains from scratch or resumes from an existing train.py checkpoint
#  * teacher-forces the first `--latent_steps` positions, then feeds each
#    HIDDEN state (after ln_f / scale_down) back as the next “token”
#  * keeps cross-entropy vs. ground-truth, with optional per-position
#    linear weighting and an initial “skip” window
# ----------------------------------------------------------------------

import argparse
import sys
import os
import time
import math
import pickle
from contextlib import nullcontext

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
recur_parser.add_argument("--resume_ckpt", default=None,
                          help="Optional checkpoint produced by train.py; omit with --init_from=scratch")
recur_parser.add_argument("--latent_steps",  type=int, default=0,
                          help="Teacher-force this many initial positions before recurrent latent feedback")
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
recur_parser.add_argument("--learn_latent_mix_alpha", default=False,
                          action=argparse.BooleanOptionalAction,
                          help="Learn the slerp blend amount as a sigmoid-constrained scalar")

# -- split cmdline -----------------------------------------------------
latent_args, remaining = recur_parser.parse_known_args()

# ----------------------------------------------------------------------
# 1-b)  now run the gigantic parser **only on the leftovers**
# ----------------------------------------------------------------------
sys.argv = [sys.argv[0]] + remaining          # fake argv for train_args
generic_args, model_group, _, _ = parse_generic_args()

# ----------------------------------------------------------------------
# 1-c)  merge both Namespaces into one `args`
# ----------------------------------------------------------------------
args = generic_args
for k, v in vars(latent_args).items():
    setattr(args, k, v)


# ----------------------------------------------------------------------
# 2)  LOAD CHECKPOINT + MODEL
# ----------------------------------------------------------------------
device = args.device
device_type = "cuda" if device.startswith("cuda") else "cpu"
dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
ptdtype = dtype_map[args.dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
if args.resume_ckpt:
    ckpt = torch.load(args.resume_ckpt, map_location=device)
    model_args = ckpt["model_args"]
else:
    if args.init_from != "scratch":
        raise ValueError("train_recurrent.py requires --resume_ckpt unless --init_from=scratch")
    meta_path = os.path.join("data", args.dataset, "meta.pkl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"scratch training requires tokenizer metadata at {meta_path}")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    model_args = {action.dest: getattr(args, action.dest) for action in model_group._group_actions}
    model_args["vocab_size"] = meta["vocab_size"]
    model_args["eval_interval"] = args.eval_interval
    ckpt = {"model_args": model_args, "iter_num": 0, "best_val_loss": float("inf")}

gpt_conf = GPTConfig(**model_args)
model = GPT(gpt_conf).to(device)

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

state_dict = unwrap_state_dict(ckpt.get("model", {}))
missing, unexpected = model.load_state_dict(state_dict, strict=False)

if missing and args.resume_ckpt:
    print(f"warning: {len(missing)} missing params (OK if all zero-grad)")
if unexpected:
    print(f"warning: {len(unexpected)} extra params ignored")

if args.compile:
    print("compiling the model (this may take a ~minute)...")
    model = torch.compile(model)

raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

# helpers exposed in patched model.py
embed_tokens = raw_model.embed_tokens
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
iter_num = ckpt["iter_num"]          # not used, but preserved

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
# 6)  PARALLEL MINI-BATCHES, EVALUATION, AND SAMPLING
# ----------------------------------------------------------------------
def get_batch(split):
    """Draw a random mini-batch, matching train.py's default sampler."""
    data = train_bin if split == "train" else val_bin
    starts = torch.randint(len(data) - block_size - 1, (args.batch_size,))
    x = torch.stack([
        torch.from_numpy(np.asarray(data[i:i + block_size], dtype=np.int64))
        for i in starts.tolist()
    ])
    y = torch.stack([
        torch.from_numpy(np.asarray(data[i + 1:i + 1 + block_size], dtype=np.int64))
        for i in starts.tolist()
    ])
    if device_type == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_val_loss():
    model.eval()
    losses = torch.zeros(args.eval_iters)
    for k in range(args.eval_iters):
        x, y = get_batch("val")
        with ctx:
            losses[k] = train_block(x, y).detach().cpu()
    model.train()
    return losses.mean().item()


def checkpoint_payload():
    return {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": ckpt["model_args"],
        "iter_num": global_step,
        "best_val_loss": best_val_loss,
        "config": vars(args),
        "latent_mix_logit": None if latent_mix_logit is None else latent_mix_logit.detach().cpu(),
    }


@torch.no_grad()
def sample_and_print():
    if not args.max_sample_tokens:
        return
    meta_path = os.path.join("data", args.dataset, "meta.pkl")
    if not os.path.exists(meta_path):
        print(f"warning: sampling skipped; {meta_path} was not found")
        return
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    if "stoi" not in meta or "itos" not in meta:
        print("warning: interval sampling currently requires a character-level meta.pkl")
        return
    encode = lambda text: [meta["stoi"][c] for c in text]
    decode = lambda ids: "".join(meta["itos"][i] for i in ids)
    start_ids = torch.tensor(encode(args.sample_start_tokens), dtype=torch.long, device=device)[None, ...]
    top_k = max(args.top_k) if isinstance(args.top_k, list) else args.top_k
    model.eval()
    with ctx:
        generated = raw_model.generate(start_ids, args.max_sample_tokens,
                                       temperature=args.temperature, top_k=top_k)
    text = decode(generated[0].tolist())
    print(f"\n--- recurrent sample @ iter {global_step} ---\n{text}\n--- end sample ---\n")
    if args.sample_file:
        sample_path = args.sample_file
        if not os.path.isabs(sample_path):
            sample_path = os.path.join(args.out_dir, sample_path)
        os.makedirs(os.path.dirname(sample_path) or ".", exist_ok=True)
        with open(sample_path, "a", encoding="utf-8") as f:
            f.write(f"\n--- iter {global_step} ---\n{text}\n")
    model.train()


# ----------------------------------------------------------------------
# 7)  ITERATION-BASED TRAINING DRIVER
# ----------------------------------------------------------------------
tb = SummaryWriter() if getattr(args, "tensorboard_log", False) else None
os.makedirs(args.out_dir, exist_ok=True)
best_ckpt_path = os.path.join(args.out_dir, "ckpt.pt")

model.train()
optimizer.zero_grad(set_to_none=True)
while global_step < args.max_iters:
    t0 = time.time()
    accumulated_loss = 0.0
    for _ in range(args.gradient_accumulation_steps):
        x, y = get_batch("train")
        with ctx:
            loss = train_block(x, y) / args.gradient_accumulation_steps
        loss.backward()
        accumulated_loss += loss.detach().item()

    if args.grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    global_step += 1

    if global_step % args.log_interval == 0:
        dt = time.time() - t0
        tokens = args.batch_size * block_size * args.gradient_accumulation_steps
        print(f"iter {global_step:>7} | loss {accumulated_loss:.4f} | "
              f"{dt * 1000:.1f} ms | {tokens / max(dt, 1e-9):.0f} tok/s")
        if tb:
            tb.add_scalar("loss/train", accumulated_loss, global_step)

    if global_step % args.eval_interval == 0 or global_step == args.max_iters:
        val_loss = estimate_val_loss()
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
        print(f"eval iter {global_step}: val loss {val_loss:.4f}")
        if tb:
            tb.add_scalar("loss/val", val_loss, global_step)
        if improved or args.always_save_checkpoint:
            torch.save(checkpoint_payload(), best_ckpt_path)
            print(f"checkpoint saved to {best_ckpt_path}")
        if improved or args.sample_each_eval:
            sample_and_print()

if tb:
    tb.flush()
    tb.close()

print("done.")
# ======================================================================
