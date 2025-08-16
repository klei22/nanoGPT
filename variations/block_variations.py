"""Block definitions and forward variations."""
from __future__ import annotations
from typing import Callable
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from variations.attention_variations import attention_dictionary
from variations.mlp_variations import get_mlp_instance
from variations.norm_variations import norm_dictionary

# type alias for the forward function
BlockForward = Callable[['Block', torch.Tensor, int, dict | None], torch.Tensor]


def parallel_mlp_forward(block, x: torch.Tensor, iter_num: int, kv_cache: dict | None = None) -> torch.Tensor:
    """Forward pass where attention and MLP run in parallel."""
    if block.use_pre_ln: # pre-LN
        x_1 = block.pre_ln(x)
    else:
        x_1 = x

    if block.use_peri_ln: # peri-LN
        attn_out = block.out_ln_attn(block._attn_forward(x_1, iter_num, kv_cache))
        mlp_out = block.out_ln_mlp(block.mlp(x_1, iter_num))
    else:
        attn_out = block._attn_forward(x_1, iter_num, kv_cache)
        mlp_out = block.mlp(x_1, iter_num)

    x = x + attn_out + mlp_out

    if block.use_post_ln: # post-LN
        x = block.post_ln(x)

    return x


def attn_then_mlp_forward(block, x: torch.Tensor, iter_num: int, kv_cache: dict | None = None) -> torch.Tensor:
    """Attention followed by MLP."""

    # Attn
    if block.use_pre_ln: # pre-LN Attn
        x_1 = block.pre_ln_attn(x)
    else:
        x_1 = x

    if block.use_peri_ln: # peri-LN Attn
        attn_out = block.out_ln_attn(block._attn_forward(x_1, iter_num, kv_cache))
    else:
        attn_out = block._attn_forward(x_1, iter_num, kv_cache)

    x = attn_out + x

    if block.use_post_ln: # post-LN Attn
        x = block.post_ln_attn(x)

    # MLP
    if block.use_pre_ln: # pre-LN MLP
        x_2 = block.pre_ln_mlp(x)
    else:
        x_2 = x

    if block.use_peri_ln: # peri-LN MLP
        mlp_out = block.out_ln_mlp(block.mlp(x_2, iter_num))
    else:
        mlp_out = block.mlp(x_2, iter_num)

    x = mlp_out + x

    if block.use_post_ln: # post-LN MLP
        x = block.post_ln_mlp(x)

    return x


block_forward_variations = {
    "parallel_mlp": parallel_mlp_forward,
    "attn_then_mlp": attn_then_mlp_forward,
}


class Block(nn.Module):
    """Transformer block supporting multiple normalization strategies."""

    def __init__(self, config, mlp=None, attn=None):
        super().__init__()

        norm_cls = norm_dictionary[config.norm_variant_attn]

        # Pre-Norm
        if config.use_pre_ln:
            if config.use_parallel_mlp:
                # parallel uses 1 less pre ln
                self.pre_ln = norm_cls(config)
            else:
                self.pre_ln_attn = norm_cls(config)
                self.pre_ln_mlp = norm_cls(config)

        # Post-Norm
        if config.use_post_ln:
            if config.use_parallel_mlp:
                # parallel uses 1 less post ln
                self.post_ln = norm_cls(config)
            else:
                self.post_ln_attn = norm_cls(config)
                self.post_ln_mlp = norm_cls(config)

        # Pero-LN
        if config.use_peri_ln:
            self.out_ln_attn = norm_cls(config)
            self.out_ln_mlp = norm_cls(config)


        self.use_pre_ln  = config.use_pre_ln
        self.use_post_ln = config.use_post_ln
        self.use_peri_ln = config.use_peri_ln

        self.use_parallel_mlp = config.use_parallel_mlp

        self.use_gradient_checkpointing = config.use_gradient_checkpointing

        # KV-cache slot id (set externally by model)
        self.kv_cache_id: int | None = None

        variant = "parallel_mlp" if self.use_parallel_mlp else "attn_then_mlp"
        self.block_forward = block_forward_variations[variant]

        if attn is None:
            self.attn = attention_dictionary[config.attention_variant](config)
        else:
            self.attn = attn

        if mlp is None:
            self.mlp = get_mlp_instance(config)
        else:
            self.mlp = mlp

    def _attn_forward(self, x: torch.Tensor, iter_num: int, kv_cache: dict | None = None):
        if kv_cache is not None:
            try:
                return self.attn(x, iter_num, kv_cache)
            except TypeError:
                return self.attn(x, iter_num)
        return self.attn(x, iter_num)

    def forward(self, x: torch.Tensor, iter_num: int, kv_cache: dict | None = None):
        if self.use_gradient_checkpointing and x.requires_grad:
            return checkpoint.checkpoint(self.block_forward, self, x, iter_num, kv_cache, use_reentrant=False)
        return self.block_forward(self, x, iter_num, kv_cache)

