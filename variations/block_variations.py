"""Block definitions and forward variations."""

from typing import Callable
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from variations.attention_variations import attention_dictionary
from variations.mlp_variations import get_mlp_instance
from variations.norm_variations import norm_dictionary

# type alias for the forward function
BlockForward = Callable[["Block", torch.Tensor, int], torch.Tensor]


def parallel_mlp_forward(block, x: torch.Tensor, iter_num: int) -> torch.Tensor:
    """Forward pass where attention and MLP run in parallel."""
    if block.use_peri_ln:
        ln_1 = block.ln_1(x)
        attn_out = block.out_ln_attn(block.attn(ln_1, iter_num))
        mlp_out = block.out_ln_mlp(block.mlp(ln_1, iter_num))
        x = x + attn_out + mlp_out
    elif block.use_post_ln:
        x = block.ln_1(x + block.attn(x, iter_num) + block.mlp(x, iter_num))
    else:  # pre-LN
        ln_1 = block.ln_1(x)
        x = x + block.attn(ln_1, iter_num) + block.mlp(ln_1, iter_num)
    return x


def attn_then_mlp_forward(block, x: torch.Tensor, iter_num: int) -> torch.Tensor:
    """Attention followed by MLP."""
    if block.use_peri_ln:
        attn_out = block.out_ln_attn(block.attn(block.ln_1(x), iter_num))
        x = x + attn_out
        mlp_out = block.out_ln_mlp(block.mlp(block.ln_2(x), iter_num))
        x = x + mlp_out
    elif block.use_post_ln:
        x = block.ln_1(x + block.attn(x, iter_num))
        x = block.ln_2(x + block.mlp(x, iter_num))
    else:  # pre-LN
        x = x + block.attn(block.ln_1(x), iter_num)
        x = x + block.mlp(block.ln_2(x), iter_num)
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
        self.ln_1 = norm_cls(config)
        if not config.use_parallel_mlp:
            self.ln_2 = norm_cls(config)

        if config.use_peri_ln:
            self.out_ln_attn = norm_cls(config)
            self.out_ln_mlp = norm_cls(config)

        self.use_post_ln = config.use_post_ln
        self.use_peri_ln = config.use_peri_ln
        self.use_parallel_mlp = config.use_parallel_mlp
        self.use_gradient_checkpointing = config.use_gradient_checkpointing

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

    def forward(self, x: torch.Tensor, iter_num: int):
        if self.use_gradient_checkpointing and x.requires_grad:
            return checkpoint.checkpoint(self.block_forward, self, x, iter_num, use_reentrant=False)
        return self.block_forward(self, x, iter_num)
