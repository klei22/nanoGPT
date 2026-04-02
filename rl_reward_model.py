"""
rl_reward_model.py

A reward model built on top of a pretrained GPT checkpoint.
Adds a scalar regression head to score generated sequences.

Usage:
  # Train a reward model on preference data
  python rl_reward_model.py --pretrained_ckpt out/ckpt.pt --out_dir out/reward

  # Or import the class directly in rl_train.py
  from rl_reward_model import RewardModel
"""

import os
import sys
import math
import argparse
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from model import GPT
from gpt_conf import GPTConfig


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------

class RewardModel(nn.Module):
    """
    Wraps a pretrained GPT and attaches a single scalar head.
    The reward is extracted from the hidden state at the last (non-padding)
    token position, then projected to a scalar via a small MLP head.

    Checkpoint contract:
        torch.save({
            'model': reward_model.state_dict(),
            'model_args': checkpoint['model_args'],   # same as pretrained
            'config': vars(args),
        }, path)
    """

    def __init__(self, gpt: GPT, n_embd: int):
        super().__init__()
        self.gpt = gpt

        # Two-layer MLP head: n_embd -> n_embd//2 -> 1
        self.reward_head = nn.Sequential(
            nn.Linear(n_embd, n_embd // 2, bias=True),
            nn.GELU(),
            nn.Linear(n_embd // 2, 1, bias=True),
        )
        # Zero-init the final layer for stable early training
        nn.init.zeros_(self.reward_head[-1].weight)
        nn.init.zeros_(self.reward_head[-1].bias)

    def forward(self, idx: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            idx:             (B, T) token indices
            attention_mask:  (B, T) 1 for real tokens, 0 for padding (optional)

        Returns:
            rewards: (B,) scalar reward per sequence
        """
        B, T = idx.shape
        device = idx.device

        # Run GPT backbone in eval-like mode (no loss computation)
        # We monkey-patch targets=None so GPT only does the forward pass
        # and returns logits for all positions.
        # We re-use the hidden states via a forward hook.
        hidden_states = {}

        def _hook(module, input, output):
            hidden_states['ln_f'] = output  # (B, T, n_embd)

        handle = self.gpt.transformer.ln_f.register_forward_hook(_hook)
        try:
            with torch.set_grad_enabled(self.training):
                self.gpt(idx, targets=None)
        finally:
            handle.remove()

        h = hidden_states['ln_f']  # (B, T, n_embd)

        # Extract the last real token's hidden state
        if attention_mask is not None:
            # last_pos[i] = index of last 1 in attention_mask[i]
            last_pos = attention_mask.long().cumsum(dim=1).argmax(dim=1)  # (B,)
        else:
            last_pos = torch.full((B,), T - 1, dtype=torch.long, device=device)

        # Gather hidden state at last real token: (B, n_embd)
        last_hidden = h[torch.arange(B, device=device), last_pos]

        rewards = self.reward_head(last_hidden).squeeze(-1)  # (B,)
        return rewards

    # ------------------------------------------------------------------
    # Preference training helpers
    # ------------------------------------------------------------------

    def preference_loss(
        self,
        chosen_idx: torch.Tensor,
        rejected_idx: torch.Tensor,
        chosen_mask: Optional[torch.Tensor] = None,
        rejected_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Bradley-Terry preference loss.
            L = -log σ(r_chosen - r_rejected)

        Args:
            chosen_idx:   (B, T) token ids for preferred responses
            rejected_idx: (B, T) token ids for rejected responses
        Returns:
            scalar loss
        """
        r_chosen   = self(chosen_idx,   chosen_mask)    # (B,)
        r_rejected = self(rejected_idx, rejected_mask)  # (B,)
        loss = -F.logsigmoid(r_chosen - r_rejected).mean()
        return loss


# ---------------------------------------------------------------------------
# Factory: load reward model from a pretrained GPT checkpoint
# ---------------------------------------------------------------------------

def load_reward_model(
    pretrained_ckpt: str,
    reward_ckpt: Optional[str] = None,
    device: str = "cpu",
    freeze_backbone: bool = False,
) -> RewardModel:
    """
    Build a RewardModel from a pretrained GPT checkpoint.

    Args:
        pretrained_ckpt:  path to a pretraining ckpt.pt
        reward_ckpt:      optional path to a previously saved reward model ckpt
        device:           torch device string
        freeze_backbone:  if True, freeze GPT weights (only train the head)

    Returns:
        RewardModel instance (on device, in train mode)
    """
    map_location = torch.device(device)
    ckpt = torch.load(pretrained_ckpt, map_location=map_location, weights_only=False)

    model_args = ckpt['model_args']
    config = GPTConfig(**model_args)
    gpt = GPT(config)
    gpt.load_state_dict(ckpt['model'])

    reward_model = RewardModel(gpt, n_embd=config.n_embd)

    if reward_ckpt is not None:
        rm_ckpt = torch.load(reward_ckpt, map_location=map_location, weights_only=False)
        reward_model.load_state_dict(rm_ckpt['model'])

    if freeze_backbone:
        for param in reward_model.gpt.parameters():
            param.requires_grad = False

    reward_model = reward_model.to(device)
    return reward_model


# ---------------------------------------------------------------------------
# Standalone training script (preference data)
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train a reward model on preference pairs")
    p.add_argument("--pretrained_ckpt", type=str, required=True,
                   help="Path to pretrained GPT checkpoint (ckpt.pt)")
    p.add_argument("--reward_ckpt", type=str, default=None,
                   help="Resume training from an existing reward model checkpoint")
    p.add_argument("--out_dir", type=str, default="out_reward",
                   help="Directory to save reward model checkpoints")
    p.add_argument("--data_dir", type=str, default="data/reward",
                   help="Directory with chosen.bin / rejected.bin preference pairs")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_iters", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Freeze GPT weights; only train the scalar head")
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_preference_data(data_dir, split="train"):
    """
    Expects flat binary files: chosen_<split>.bin and rejected_<split>.bin
    Each file contains token ids as uint16, one sequence per row of shape (seq_len,).
    Returns (chosen_tokens, rejected_tokens) as int64 tensors.
    """
    import numpy as np
    chosen   = np.fromfile(os.path.join(data_dir, f"chosen_{split}.bin"),  dtype=np.uint16)
    rejected = np.fromfile(os.path.join(data_dir, f"rejected_{split}.bin"), dtype=np.uint16)
    chosen   = torch.from_numpy(chosen.astype(np.int64))
    rejected = torch.from_numpy(rejected.astype(np.int64))
    return chosen, rejected


def get_batch(chosen_flat, rejected_flat, seq_len, batch_size, device):
    """Sample a batch of (chosen, rejected) pairs at random."""
    n = len(chosen_flat) // seq_len
    idx = torch.randint(n, (batch_size,))
    chosen   = torch.stack([chosen_flat  [i * seq_len:(i + 1) * seq_len] for i in idx]).to(device)
    rejected = torch.stack([rejected_flat[i * seq_len:(i + 1) * seq_len] for i in idx]).to(device)
    return chosen, rejected


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading reward model from {args.pretrained_ckpt}")
    rm = load_reward_model(
        args.pretrained_ckpt,
        reward_ckpt=args.reward_ckpt,
        device=args.device,
        freeze_backbone=args.freeze_backbone,
    )
    rm.train()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, rm.parameters()),
        lr=args.lr,
    )

    # Load preference pairs
    chosen_train, rejected_train = load_preference_data(args.data_dir, "train")
    chosen_val,   rejected_val   = load_preference_data(args.data_dir, "val")

    # Infer sequence length from GPT config
    seq_len = rm.gpt.config.block_size

    best_val_loss = float('inf')
    for it in range(1, args.max_iters + 1):
        chosen, rejected = get_batch(chosen_train, rejected_train, seq_len, args.batch_size, args.device)

        optimizer.zero_grad()
        loss = rm.preference_loss(chosen, rejected)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
        optimizer.step()

        if it % args.eval_interval == 0:
            rm.eval()
            with torch.no_grad():
                c_val, r_val = get_batch(chosen_val, rejected_val, seq_len, args.batch_size * 4, args.device)
                val_loss = rm.preference_loss(c_val, r_val).item()

            # Accuracy: how often r_chosen > r_rejected
            r_c = rm(c_val)
            r_r = rm(r_val)
            acc = (r_c > r_r).float().mean().item()

            print(f"iter {it:5d} | train loss {loss.item():.4f} | val loss {val_loss:.4f} | acc {acc:.3f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save in a format compatible with load_reward_model()
                ckpt = torch.load(args.pretrained_ckpt, map_location='cpu', weights_only=False)
                torch.save({
                    'model': rm.state_dict(),
                    'model_args': ckpt['model_args'],
                    'val_loss': val_loss,
                    'iter_num': it,
                    'config': vars(args),
                }, os.path.join(args.out_dir, 'reward_ckpt.pt'))
                print(f"  -> saved reward_ckpt.pt (val_loss={val_loss:.4f})")

            rm.train()

    print("Reward model training complete.")


if __name__ == "__main__":
    main()
