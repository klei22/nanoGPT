"""Block definitions and forward variations."""
from __future__ import annotations
from typing import Callable
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from variations.attention_variations import attention_dictionary
from variations.mlp_variations import get_mlp_instance
from variations.norm_variations import norm_dictionary
from variations.learned_confidence_variations import learned_confidence_dictionary

# type alias for the forward function
BlockForward = Callable[['Block', torch.Tensor, int], torch.Tensor]


def parallel_mlp_forward(block, x: torch.Tensor, iter_num: int) -> torch.Tensor:
    """Forward pass where attention and MLP run in parallel."""
    if block.use_pre_ln: # pre-LN
        x_1 = block.pre_ln(x)
    else:
        x_1 = x

    if block.use_peri_ln: # peri-LN
        attn_out = block.out_ln_attn(block.attn(x_1, iter_num))
        mlp_out = block.out_ln_mlp(block.mlp(x_1, iter_num))
    else:
        attn_out = block.attn(x_1, iter_num)
        mlp_out = block.mlp(x_1, iter_num)

    if block.attn_resid_scaler is not None:
        attn_out = block.attn_resid_scaler(attn_out)
    if block.mlp_resid_scaler is not None:
        mlp_out = block.mlp_resid_scaler(mlp_out)

    x = x + attn_out + mlp_out

    if block.use_post_ln: # post-LN
        x = block.post_ln(x)

    return x


def attn_then_mlp_forward(block, x: torch.Tensor, iter_num: int) -> torch.Tensor:
    """Attention followed by MLP."""

    # Attn
    if block.use_pre_ln: # pre-LN Attn
        x_1 = block.pre_ln_attn(x)
    else:
        x_1 = x

    if block.use_peri_ln: # peri-LN Attn
        attn_out = block.out_ln_attn(block.attn(x_1, iter_num))
    else:
        attn_out = block.attn(x_1, iter_num)

    if block.attn_resid_scaler is not None:
        attn_out = block.attn_resid_scaler(attn_out)

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

    if block.mlp_resid_scaler is not None:
        mlp_out = block.mlp_resid_scaler(mlp_out)

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
        self.recompute_backward_pass = getattr(config, 'recompute_backward_pass', False)

        if config.use_attn_resid_scaling:
            cls = learned_confidence_dictionary[config.attn_confidence_variant]
            self.attn_resid_scaler = cls(config, prefix="attn")
        else:
            self.attn_resid_scaler = None
        if config.use_mlp_resid_scaling:
            cls = learned_confidence_dictionary[config.mlp_confidence_variant]
            self.mlp_resid_scaler = cls(config, prefix="mlp")
        else:
            self.mlp_resid_scaler = None

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
            use_reentrant = not self.recompute_backward_pass
            return checkpoint.checkpoint(
                lambda inp: self.block_forward(self, inp, iter_num),
                x,
                use_reentrant=use_reentrant,
            )
        return self.block_forward(self, x, iter_num)

