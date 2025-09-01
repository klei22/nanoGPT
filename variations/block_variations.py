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
from quantization.quantize import fake_quantize_act

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


def edgellm_asic_forward(block, x: torch.Tensor, iter_num: int) -> torch.Tensor:
    """EdgeLLM ASIC forward: Attention followed by MLP with skip connection accumulation between blocks."""

    x_residual = x

    if block.quantization_dict["quantize_asic_prenorm"]:
        num_bits = block.quantization_dict["quantize_asic_bits"]
        quant_method = block.quantization_dict["activations_quant_method"]
        x = fake_quantize_act(block, "asic_attn_prenorm", x, num_bits, quant_method, iter_num)

    # Attn
    if block.use_pre_ln: # pre-LN Attn
        x_1 = block.pre_ln_attn(x)
    else:
        x_1 = x

    attn_out = block.attn(x_1, iter_num)

    if block.attn_resid_scaler is not None:
        attn_out = block.attn_resid_scaler(attn_out)

    x = attn_out + x

    if block.quantization_dict["quantize_asic_prenorm"]:
        num_bits = block.quantization_dict["quantize_asic_bits"]
        quant_method = block.quantization_dict["activations_quant_method"]
        x = fake_quantize_act(block, "asic_mlp_prenorm", x, num_bits, quant_method, iter_num)

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

    x = x_residual + x

    return x


block_forward_variations = {
    "parallel_mlp": parallel_mlp_forward,
    "attn_then_mlp": attn_then_mlp_forward,
    "edgellm_asic": edgellm_asic_forward
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
        self.use_edgellm_asic = config.use_edgellm_asic

        self.use_gradient_checkpointing = config.use_gradient_checkpointing

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

        if self.use_parallel_mlp:
            variant = "parallel_mlp"
        elif self.use_edgellm_asic:
            variant = "edgellm_asic"
        else:
            variant = "attn_then_mlp"

        if self.use_edgellm_asic:
            self.quantization_dict = {}
            self.quantization_dict["quantize_asic_prenorm"] = config.quantize_asic_prenorm
            self.quantization_dict["quantize_asic_bits"] = config.quantize_asic_bits
            self.quantization_dict["activations_quant_method"] = config.activations_quant_method
            self.full_quant_iteration = config.full_quant_iteration
            self.eval_interval = config.eval_interval
            self.start_quant_level = config.start_quant_level
            self.quant_scheduler = config.quant_scheduler

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

