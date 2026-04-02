"""
rl_train.py

Reinforcement Learning fine-tuning of a pretrained nanoGPT model.

Algorithm: REINFORCE with a KL-divergence penalty against the frozen reference
model (the original pretrained weights).  This is the simplest approach that
actually works for language models and is equivalent to the "RLHF with KL
control" formulation used in early InstructGPT work.

Pipeline overview
-----------------
1. Load pretrained policy  (from ckpt.pt)
2. Clone it as a frozen reference model
3. For each training step:
   a. Sample a batch of prompt token sequences from disk
   b. Generate completions with the *policy* (differentiable log-probs kept)
   c. Score the full (prompt + completion) with the reward model
   d. Compute per-token advantages = reward - baseline (running mean)
   e. Policy gradient loss = -mean( advantages * log_probs_of_completion )
   f. KL penalty = beta * KL( policy || reference ) over completion tokens
   g. total_loss = pg_loss + kl_loss
   h. Gradient step, save checkpoint in standard ckpt.pt format

Usage example
-------------
python rl_train.py \
    --pretrained_ckpt out/ckpt.pt \
    --reward_ckpt     out_reward/reward_ckpt.pt \
    --data_dir        data/prompts \
    --out_dir         out_rl \
    --max_iters       500 \
    --kl_coef         0.1 \
    --gen_len         64 \
    --batch_size      8
"""

import os
import sys
import argparse
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.nn import functional as F

from model import GPT
from gpt_conf import GPTConfig
from rl_reward_model import RewardModel, load_reward_model


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="RL fine-tuning of a pretrained nanoGPT model")

    # Checkpoint paths
    p.add_argument("--pretrained_ckpt", type=str, required=True,
                   help="Path to pretrained GPT checkpoint (ckpt.pt)")
    p.add_argument("--reward_ckpt", type=str, default=None,
                   help="Path to trained reward model checkpoint. "
                        "If omitted, a simple length-based toy reward is used.")
    p.add_argument("--resume_rl_ckpt", type=str, default=None,
                   help="Path to a previously saved RL checkpoint to resume from")
    p.add_argument("--out_dir", type=str, default="out_rl")

    # Data
    p.add_argument("--data_dir", type=str, default="data/prompts",
                   help="Directory with prompt token files (prompts_train.bin, prompts_val.bin)")
    p.add_argument("--prompt_len", type=int, default=64,
                   help="Number of tokens to use as the prompt prefix")
    p.add_argument("--gen_len", type=int, default=64,
                   help="Number of new tokens to generate per rollout")

    # Training
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_iters", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_interval", type=int, default=50)

    # RL-specific
    p.add_argument("--kl_coef", type=float, default=0.1,
                   help="Coefficient for KL divergence penalty term")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature for rollout generation")
    p.add_argument("--top_k", type=int, default=50,
                   help="Top-k filtering for rollout generation (0 = disabled)")
    p.add_argument("--normalize_rewards", action="store_true", default=True,
                   help="Normalize rewards to zero mean / unit variance per batch")
    p.add_argument("--baseline_momentum", type=float, default=0.99,
                   help="EMA momentum for the running reward baseline")

    # System
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="float32",
                   choices=["float32", "bfloat16", "float16"])
    p.add_argument("--compile", action="store_true",
                   help="Use torch.compile for the policy model")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint helpers (matches the existing ckpt.pt schema)
# ---------------------------------------------------------------------------

def load_policy_from_ckpt(path: str, device: str):
    """
    Load a GPT policy model from any standard ckpt.pt produced by train.py
    or a previous rl_train.py run.  Returns (model, checkpoint_dict).
    """
    ckpt = torch.load(path, map_location=torch.device(device), weights_only=False)
    model_args = ckpt['model_args']
    config = GPTConfig(**model_args)
    model = GPT(config)
    model.load_state_dict(ckpt['model'])
    model = model.to(device)
    return model, ckpt


def save_rl_checkpoint(
    model: GPT,
    optimizer,
    iter_num: int,
    best_val_reward: float,
    original_ckpt: dict,
    args,
    out_dir: str,
    filename: str = "ckpt.pt",
):
    """
    Save an RL checkpoint in the same schema as train.py so it can be loaded
    by sample.py or resumed with either train.py or rl_train.py.
    """
    os.makedirs(out_dir, exist_ok=True)
    # Reuse model_args and config from the original checkpoint so all
    # architecture flags are preserved.
    ckpt = {
        'model':          model.state_dict(),
        'optimizer':      optimizer.state_dict(),
        'model_args':     original_ckpt['model_args'],
        'iter_num':       iter_num,
        'best_val_loss':  -best_val_reward,   # negate so lower=better convention holds
        'best_iter':      iter_num,
        'best_tokens':    original_ckpt.get('best_tokens', 0),
        'config':         vars(args),
        # Extra RL metadata (informational, not used by train.py)
        'rl_iter':        iter_num,
        'best_val_reward': best_val_reward,
    }
    path = os.path.join(out_dir, filename)
    torch.save(ckpt, path)
    return path


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_prompts(data_dir: str, split: str = "train"):
    """
    Load flat uint16 token file.  Returns a 1-D int64 tensor of all tokens.
    """
    import numpy as np
    fname = os.path.join(data_dir, f"prompts_{split}.bin")
    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"Prompt file not found: {fname}\n"
            "Create it by running your tokeniser and saving with:\n"
            "  arr.astype(np.uint16).tofile(fname)"
        )
    data = torch.from_numpy(
        __import__("numpy").fromfile(fname, dtype=__import__("numpy").uint16).astype("int64")
    )
    return data


def sample_prompts(data: torch.Tensor, prompt_len: int, batch_size: int, device: str):
    """
    Sample a batch of random prompt windows from the flat token array.
    Returns (B, prompt_len) tensor.
    """
    n = len(data) - prompt_len
    starts = torch.randint(n, (batch_size,))
    prompts = torch.stack([data[s:s + prompt_len] for s in starts]).to(device)
    return prompts


# ---------------------------------------------------------------------------
# Generation with per-token log-probabilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_with_ref_logprobs(
    policy: GPT,
    ref_model: GPT,
    prompt_ids: torch.Tensor,
    gen_len: int,
    temperature: float = 1.0,
    top_k: int = 50,
):
    """
    Auto-regressively generate `gen_len` tokens from the policy while also
    computing the reference model's log-probabilities at each step.

    Returns:
        generated_ids:  (B, gen_len) int64 – only the new tokens
        policy_logprobs: (B, gen_len) float – log p_policy(token_t | context)
        ref_logprobs:    (B, gen_len) float – log p_ref(token_t | context)
    """
    B = prompt_ids.shape[0]
    device = prompt_ids.device
    block_size = policy.config.block_size

    ctx = torch.cat([prompt_ids], dim=1)  # (B, prompt_len)

    gen_ids    = []
    pol_lps    = []
    ref_lps    = []

    for _ in range(gen_len):
        ctx_crop = ctx[:, -block_size:]

        # Policy logits
        pol_logits, _ = policy(ctx_crop)          # (B, 1, V)  (inference path)
        pol_logits = pol_logits[:, -1, :]         # (B, V)

        # Reference logits (frozen)
        ref_logits, _ = ref_model(ctx_crop)
        ref_logits = ref_logits[:, -1, :]

        # Apply temperature
        if temperature != 1.0:
            pol_logits = pol_logits / temperature

        # Top-k filtering
        if top_k > 0:
            v, _ = torch.topk(pol_logits, min(top_k, pol_logits.size(-1)))
            pol_logits[pol_logits < v[:, [-1]]] = float('-inf')

        # Sample
        probs = F.softmax(pol_logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)  # (B, 1)

        # Log-probs under policy and reference
        pol_lp = F.log_softmax(pol_logits, dim=-1)
        ref_lp = F.log_softmax(ref_logits, dim=-1)  # (B, V)

        pol_lps.append(pol_lp.gather(1, next_tok).squeeze(1))   # (B,)
        ref_lps.append(ref_lp.gather(1, next_tok).squeeze(1))   # (B,)
        gen_ids.append(next_tok.squeeze(1))

        ctx = torch.cat([ctx, next_tok], dim=1)

    generated_ids    = torch.stack(gen_ids, dim=1)    # (B, gen_len)
    policy_logprobs  = torch.stack(pol_lps, dim=1)    # (B, gen_len)
    ref_logprobs_out = torch.stack(ref_lps, dim=1)    # (B, gen_len)

    return generated_ids, policy_logprobs, ref_logprobs_out


def recompute_policy_logprobs(
    policy: GPT,
    prompt_ids: torch.Tensor,
    gen_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Recompute log-probs of the generated tokens under the *current* policy
    (with gradients).  Used in the gradient-update step.

    Returns:
        logprobs: (B, gen_len) float with gradients
    """
    B, gen_len = gen_ids.shape
    block_size = policy.config.block_size

    # Full sequence for teacher forcing: (B, prompt_len + gen_len)
    full_seq = torch.cat([prompt_ids, gen_ids], dim=1)
    # Input is full_seq[:-1], targets are full_seq[1:]
    inp  = full_seq[:, :-1]
    tgt  = full_seq[:, 1:]

    inp_crop = inp[:, -block_size:]
    tgt_crop = tgt[:, -(inp_crop.shape[1]):]

    logits, _ = policy(inp_crop, targets=None)   # (B, T, V)

    # Slice only the positions corresponding to generated tokens
    # Positions of gen tokens in the output:  last gen_len positions
    gen_logits = logits[:, -gen_len:, :]          # (B, gen_len, V)

    log_probs_all = F.log_softmax(gen_logits, dim=-1)  # (B, gen_len, V)
    # Gather log-prob of the token that was actually sampled
    lp = log_probs_all.gather(2, gen_ids.unsqueeze(-1)).squeeze(-1)  # (B, gen_len)
    return lp


# ---------------------------------------------------------------------------
# Toy reward function (used when no reward model is provided)
# ---------------------------------------------------------------------------

def toy_reward(prompt_ids: torch.Tensor, gen_ids: torch.Tensor) -> torch.Tensor:
    """
    Placeholder: reward = fraction of unique tokens in the generated sequence.
    Replace with your actual reward signal.

    Returns:
        rewards: (B,) float tensor
    """
    B, L = gen_ids.shape
    rewards = []
    for b in range(B):
        unique_frac = gen_ids[b].unique().numel() / L
        rewards.append(unique_frac)
    return torch.tensor(rewards, dtype=torch.float32, device=gen_ids.device)


# ---------------------------------------------------------------------------
# Running reward baseline (simple EMA)
# ---------------------------------------------------------------------------

class RewardBaseline:
    """Exponential moving average baseline to reduce policy gradient variance."""

    def __init__(self, momentum: float = 0.99):
        self.momentum = momentum
        self.value = 0.0
        self.initialized = False

    def update(self, rewards: torch.Tensor) -> float:
        mean_r = rewards.mean().item()
        if not self.initialized:
            self.value = mean_r
            self.initialized = True
        else:
            self.value = self.momentum * self.value + (1 - self.momentum) * mean_r
        return self.value

    def get(self) -> float:
        return self.value


# ---------------------------------------------------------------------------
# Main RL training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device

    # ------------------------------------------------------------------
    # 1. Load policy model
    # ------------------------------------------------------------------
    print(f"Loading policy from: {args.pretrained_ckpt}")
    if args.resume_rl_ckpt:
        policy, original_ckpt = load_policy_from_ckpt(args.resume_rl_ckpt, device)
        start_iter = original_ckpt.get('rl_iter', 0) + 1
        print(f"  Resuming from RL iter {start_iter}")
    else:
        policy, original_ckpt = load_policy_from_ckpt(args.pretrained_ckpt, device)
        start_iter = 1

    policy.train()

    # ------------------------------------------------------------------
    # 2. Frozen reference model (same weights as original pretrained ckpt)
    # ------------------------------------------------------------------
    print("Cloning reference model (frozen)")
    ref_model, _ = load_policy_from_ckpt(args.pretrained_ckpt, device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # ------------------------------------------------------------------
    # 3. Reward model (or toy reward)
    # ------------------------------------------------------------------
    if args.reward_ckpt:
        print(f"Loading reward model from: {args.reward_ckpt}")
        reward_model = load_reward_model(
            args.pretrained_ckpt,
            reward_ckpt=args.reward_ckpt,
            device=device,
            freeze_backbone=True,
        )
        reward_model.eval()
        for param in reward_model.parameters():
            param.requires_grad = False

        def score_fn(prompt_ids, gen_ids):
            full = torch.cat([prompt_ids, gen_ids], dim=1)
            return reward_model(full)
    else:
        print("No reward model provided – using toy reward (unique token fraction)")
        def score_fn(prompt_ids, gen_ids):
            return toy_reward(prompt_ids, gen_ids)

    # ------------------------------------------------------------------
    # 4. Optimizer
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    if args.resume_rl_ckpt and 'optimizer' in original_ckpt:
        try:
            optimizer.load_state_dict(original_ckpt['optimizer'])
        except Exception:
            print("  Warning: could not restore optimizer state; starting fresh")

    if args.compile:
        print("Compiling policy with torch.compile …")
        policy = torch.compile(policy)

    # ------------------------------------------------------------------
    # 5. Data
    # ------------------------------------------------------------------
    print(f"Loading prompt data from: {args.data_dir}")
    train_data = load_prompts(args.data_dir, "train")
    val_data   = load_prompts(args.data_dir, "val")

    baseline = RewardBaseline(momentum=args.baseline_momentum)
    best_val_reward = float('-inf')

    ctx = (
        torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16)
        if args.dtype == "bfloat16" and device.startswith("cuda")
        else nullcontext()
    )

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    print(f"\nStarting RL training for {args.max_iters} iters "
          f"(kl_coef={args.kl_coef}, temp={args.temperature})")

    for it in range(start_iter, args.max_iters + 1):

        # ---- 6a. Sample prompts and generate rollouts -----------------
        prompt_ids = sample_prompts(train_data, args.prompt_len, args.batch_size, device)

        with torch.no_grad():
            gen_ids, _, ref_lps = generate_with_ref_logprobs(
                policy, ref_model,
                prompt_ids,
                gen_len=args.gen_len,
                temperature=args.temperature,
                top_k=args.top_k,
            )

        # ---- 6b. Score with reward model ------------------------------
        with torch.no_grad():
            rewards = score_fn(prompt_ids, gen_ids).float()   # (B,)

        # ---- 6c. Normalise rewards and compute advantages -------------
        if args.normalize_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        baseline_val = baseline.update(rewards)
        advantages = rewards - baseline_val                  # (B,)

        # ---- 6d. Recompute policy log-probs WITH gradients -----------
        with ctx:
            pol_lps = recompute_policy_logprobs(policy, prompt_ids, gen_ids)  # (B, gen_len)

        # ---- 6e. Policy gradient loss  (REINFORCE) -------------------
        #   L_pg = -E[ advantage * sum_t log π(a_t|s_t) ]
        sum_lp = pol_lps.sum(dim=1)                          # (B,)
        pg_loss = -(advantages.detach() * sum_lp).mean()

        # ---- 6f. KL penalty  -----------------------------------------
        #   KL(policy || ref) ≈ log π - log π_ref  (per token, then mean)
        kl_per_token = pol_lps - ref_lps.detach()            # (B, gen_len)
        kl_loss = kl_per_token.mean()

        total_loss = pg_loss + args.kl_coef * kl_loss

        # ---- 6g. Gradient step ----------------------------------------
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        optimizer.step()

        # ---- 6h. Logging ----------------------------------------------
        if it % max(1, args.eval_interval // 10) == 0:
            print(
                f"iter {it:5d} | loss {total_loss.item():.4f} "
                f"| pg {pg_loss.item():.4f} | kl {kl_loss.item():.4f} "
                f"| reward {rewards.mean().item():.4f} "
                f"| baseline {baseline_val:.4f}"
            )

        # ---- 6i. Validation and checkpoint ----------------------------
        if it % args.eval_interval == 0:
            policy.eval()
            with torch.no_grad():
                val_prompts = sample_prompts(
                    val_data, args.prompt_len, args.batch_size * 2, device
                )
                val_gen, _, _ = generate_with_ref_logprobs(
                    policy, ref_model, val_prompts,
                    gen_len=args.gen_len,
                    temperature=1.0, top_k=args.top_k,
                )
                val_rewards = score_fn(val_prompts, val_gen).float()
                mean_val_reward = val_rewards.mean().item()

            print(f"  [VAL] iter {it} | val_reward {mean_val_reward:.4f}")

            if mean_val_reward > best_val_reward:
                best_val_reward = mean_val_reward
                path = save_rl_checkpoint(
                    policy if not args.compile else policy._orig_mod,
                    optimizer, it, best_val_reward,
                    original_ckpt, args, args.out_dir,
                    filename="ckpt.pt",
                )
                print(f"  -> saved best checkpoint: {path} (val_reward={best_val_reward:.4f})")

            # Also save a timestamped checkpoint
            save_rl_checkpoint(
                policy if not args.compile else policy._orig_mod,
                optimizer, it, mean_val_reward,
                original_ckpt, args, args.out_dir,
                filename=f"ckpt_rl_{it:06d}.pt",
            )

            policy.train()

    # Final checkpoint
    save_rl_checkpoint(
        policy if not args.compile else policy._orig_mod,
        optimizer, args.max_iters, best_val_reward,
        original_ckpt, args, args.out_dir,
        filename="ckpt_rl_final.pt",
    )
    print(f"\nRL training complete. Checkpoints saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
