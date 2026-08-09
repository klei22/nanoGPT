"""
nanogpt_kvcache.py

GPT-2 model with a pre-allocated KV-cache, written to be compatible with
torch.export / ExecuTorch.

Two export targets are supported:

  prefill(tokens)           – process the entire prompt at once; fills the
                              KV-cache side-by-side.  Returns logits for every
                              position so the caller can sample the FIRST
                              generated token.  This is the "Time-to-First-Token"
                              critical path.

  decode(token, input_pos)  – process exactly ONE new token using the already-
                              filled KV-cache.  Returns logits for that single
                              position.  Repeated N times to produce N tokens.

Key design choices
------------------
* KV-cache buffers are module-level nn.Parameters registered with
  register_buffer().  torch.export serialises them into the .pte file so
  on-device state persists across Module::forward() calls.
* In-place slice writes (`cache[..., pos:pos+T, :] = new_kv`) are used for
  the update.  ExecuTorch's ExecATen executor supports copy_ semantics on
  buffers, so these mutations are preserved between invocations.
* Attention is always MATH SDPA (no Flash-Attention) so it lowers cleanly
  through the ExecuTorch edge dialects.
* Dropout is disabled (eval mode) for export.
"""

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """LayerNorm with optional bias (PyTorch built-in doesn't allow bias=False)."""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class KVCache(nn.Module):
    """
    Pre-allocated key/value cache for one transformer layer.

    Stored as plain buffers so ExecuTorch serialises them into the .pte
    and mutations are visible across invocations.
    """

    def __init__(self, max_batch: int, max_seq: int, n_head: int, head_dim: int):
        super().__init__()
        shape = (max_batch, n_head, max_seq, head_dim)
        self.register_buffer("k", torch.zeros(shape))
        self.register_buffer("v", torch.zeros(shape))

    def update(
        self,
        k_new: torch.Tensor,   # (B, n_head, T, head_dim)
        v_new: torch.Tensor,
        start_pos: int,
        seq_len:   int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Write new KV slices and return the full accumulated KV tensors."""
        # In-place write – tracked as buffer mutation by ExecuTorch
        self.k[:, :, start_pos : start_pos + seq_len, :] = k_new
        self.v[:, :, start_pos : start_pos + seq_len, :] = v_new
        # Return everything up to current position
        return (
            self.k[:, :, : start_pos + seq_len, :],
            self.v[:, :, : start_pos + seq_len, :],
        )


class CausalSelfAttentionKV(nn.Module):
    """Multi-head causal self-attention with an external KVCache."""

    def __init__(self, config: GPTConfig, kv_cache: KVCache):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn   = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj   = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head   = config.n_head
        self.n_embd   = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.kv_cache = kv_cache

    def forward(
        self,
        x:         torch.Tensor,  # (B, T, C)
        start_pos: int,
        is_causal: bool,
    ) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        k_full, v_full = self.kv_cache.update(k, v, start_pos, T)

        # MATH backend (no flash) – works on all ExecuTorch targets
        y = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu   = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.gelu(self.c_fc(x)))


class BlockKV(nn.Module):
    def __init__(self, config: GPTConfig, kv_cache: KVCache):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttentionKV(config, kv_cache)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(
        self,
        x:         torch.Tensor,
        start_pos: int,
        is_causal: bool,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), start_pos, is_causal)
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class GPTWithKVCache(nn.Module):
    """
    nanoGPT with a pre-allocated KV-cache.

    Forward signatures
    ------------------
    prefill(tokens)
        tokens : LongTensor (1, prompt_len)
        Returns logits (1, prompt_len, vocab_size).
        Fills the KV-cache for positions [0, prompt_len).

    decode(token, start_pos)
        token     : LongTensor (1, 1)
        start_pos : int – position index of this new token
        Returns logits (1, 1, vocab_size).
        Updates the KV-cache at position start_pos.

    For ExecuTorch export call forward() directly with an extra bool flag:
        forward(tokens, start_pos, is_causal)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        head_dim    = config.n_embd // config.n_head

        # Token + position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)

        # One KVCache per layer, stored as sub-modules so buffers are tracked
        self.kv_caches = nn.ModuleList([
            KVCache(1, config.block_size, config.n_head, head_dim)
            for _ in range(config.n_layer)
        ])

        self.blocks = nn.ModuleList([
            BlockKV(config, self.kv_caches[i])
            for i in range(config.n_layer)
        ])

        self.ln_f    = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying
        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    # ------------------------------------------------------------------
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    def forward(
        self,
        tokens:    torch.Tensor,  # (1, T)   LongTensor
        start_pos: int,            # first position index for this batch of tokens
        is_causal: bool,           # True for prefill, False for single-token decode
    ) -> torch.Tensor:             # (1, T, vocab_size)
        B, T = tokens.shape
        pos = torch.arange(start_pos, start_pos + T, dtype=torch.long, device=tokens.device)

        tok_emb = self.wte(tokens)          # (1, T, n_embd)
        pos_emb = self.wpe(pos).unsqueeze(0)  # (1, T, n_embd)
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x, start_pos, is_causal)

        x = self.ln_f(x)
        logits = self.lm_head(x)   # (1, T, vocab_size)
        return logits

    # ------------------------------------------------------------------
    # Convenience wrappers (used at Python inference time, not for export)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, tokens: torch.Tensor) -> torch.Tensor:
        """Process prompt.  Returns (1, T, vocab_size)."""
        return self.forward(tokens, start_pos=0, is_causal=True)

    @torch.no_grad()
    def decode(self, token: torch.Tensor, start_pos: int) -> torch.Tensor:
        """Process one new token.  Returns (1, 1, vocab_size)."""
        return self.forward(token, start_pos=start_pos, is_causal=False)

    # ------------------------------------------------------------------
    # Weight loading from HuggingFace GPT-2 checkpoint
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, model_type: str = "gpt2") -> "GPTWithKVCache":
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        config_map = {
            "gpt2":        GPTConfig(n_layer=12, n_head=12, n_embd=768,  vocab_size=50257, block_size=1024, bias=True),
            "gpt2-medium": GPTConfig(n_layer=24, n_head=16, n_embd=1024, vocab_size=50257, block_size=1024, bias=True),
            "gpt2-large":  GPTConfig(n_layer=36, n_head=20, n_embd=1280, vocab_size=50257, block_size=1024, bias=True),
            "gpt2-xl":     GPTConfig(n_layer=48, n_head=25, n_embd=1600, vocab_size=50257, block_size=1024, bias=True),
        }
        config = config_map[model_type]
        model  = cls(config)

        from transformers import GPT2LMHeadModel
        print(f"Loading weights from HuggingFace: {model_type}")
        hf_model = GPT2LMHeadModel.from_pretrained(model_type)
        hf_sd    = hf_model.state_dict()

        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        sd = model.state_dict()
        for k_hf, v_hf in hf_sd.items():
            # Map HF keys → our keys
            k_ours = (
                k_hf
                .replace("transformer.h.",   "blocks.")
                .replace(".attn.c_attn",      ".attn.c_attn")
                .replace(".attn.c_proj",      ".attn.c_proj")
                .replace(".mlp.",             ".mlp.")
                .replace(".ln_1",             ".ln_1")
                .replace(".ln_2",             ".ln_2")
                .replace("transformer.ln_f",  "ln_f")
                .replace("transformer.wte",   "wte")
                .replace("transformer.wpe",   "wpe")
                .replace("lm_head",           "lm_head")
            )
            if k_ours not in sd:
                continue
            if any(k_hf.endswith(t) for t in transposed):
                with torch.no_grad():
                    sd[k_ours].copy_(v_hf.t())
            else:
                with torch.no_grad():
                    sd[k_ours].copy_(v_hf)

        model.load_state_dict(sd)
        return model
