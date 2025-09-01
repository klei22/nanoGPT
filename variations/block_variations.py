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

# -----------------------
# Block Forward Variations
# -----------------------

def parallel_mlp_forward(block, x: torch.Tensor, iter_num: int) -> torch.Tensor:
    """Forward pass where attention and MLP run in parallel."""

    # Make sure not to override skip connection
    x_in = x

    # Pre-LN
    if block.use_pre_ln:
        x_in = block.pre_ln(x_in)

    # Perform Operations
    attn_out = block.attn(x_in, iter_num)
    mlp_out = block.mlp(x_in, iter_num)

    # Peri-LN
    if block.use_peri_ln:
        attn_out = block.peri_ln_attn(attn_out)
        mlp_out = block.peri_ln_mlp(mlp_out)

    # MLP and Attn Output Scaling
    if block.attn_resid_scaler is not None:
        attn_out = block.attn_resid_scaler(attn_out)
    if block.mlp_resid_scaler is not None:
        mlp_out = block.mlp_resid_scaler(mlp_out)

    # Skip Connection
    x = x + attn_out + mlp_out

    # Post-LN
    if block.use_post_ln:
        x = block.post_ln(x)

    return x


def attn_then_mlp_forward(block, x: torch.Tensor, iter_num: int) -> torch.Tensor:
    """Attention followed by MLP."""

    # Make sure not to override skip connection
    x_attn_in = x

    # Attn Pre-LN
    if block.use_pre_ln:
        x_attn_in = block.pre_ln_attn(x_attn_in)

    # Attn Operation
    attn_out = block.attn(x_attn_in, iter_num)

    # Attn Peri-LN
    if block.use_peri_ln:
        attn_out = block.out_ln_attn(attn_out)

    # Attn Output Scaling
    if block.attn_resid_scaler is not None:
        attn_out = block.attn_resid_scaler(attn_out)

    # Attn Skip Connection
    x = attn_out + x

    # Attn Post-LN
    if block.use_post_ln:
        x = block.post_ln_attn(x)

    # Make sure not to override skip connection
    x_mlp_in = x

    # MLP Pre-LN
    if block.use_pre_ln:
        x_mlp_in = block.pre_ln_mlp(x_mlp_in)

    # MLP Operation
    mlp_out = block.mlp(x_mlp_in, iter_num)

    # MLP Peri-LN
    if block.use_peri_ln:
        mlp_out = block.out_ln_mlp(mlp_out)

    # MLP Output Scaling
    if block.mlp_resid_scaler is not None:
        mlp_out = block.mlp_resid_scaler(mlp_out)

    # MLP Skip Connection
    x = mlp_out + x

    # MLP Post-LN
    if block.use_post_ln:
        x = block.post_ln_mlp(x)

    return x


def edgellm_asic_forward(block, x: torch.Tensor, iter_num: int) -> torch.Tensor:
    """EdgeLLM ASIC forward: Attention followed by MLP with skip connection accumulation between blocks."""

    # Separate Full Precision Residual 'x' from "x_quantized_residual'
    x_quantized_residual = x

    # Quantize x_attn_in before pre-norm
    if block.quantization_dict["quantize_asic_prenorm"]:
        num_bits = block.quantization_dict["quantize_asic_bits"]
        quant_method = block.quantization_dict["activations_quant_method"]
        x_quantized_residual = fake_quantize_act(block, "asic_attn_prenorm", x_quantized_residual, num_bits, quant_method, iter_num)

    # Store Original Quantized Residual for Later
    # Propagate only x_quantized_residual on-chip
    x_quantized_residual_initial = x_quantized_residual

    # On-Chip: Input Quantized Residual to Chip

    # Attn Pre-LN
    x_attn_in = x_quantized_residual
    if block.use_pre_ln:
        x_attn_in = block.pre_ln_attn(x_attn_in)

    # Attn Operation
    attn_out = block.attn(x_attn_in, iter_num)

    # Attn Peri-LN
    if block.use_peri_ln:
        attn_out = block.out_ln_attn(attn_out)

    # Attn Output Scaling
    if block.attn_resid_scaler is not None:
        attn_out = block.attn_resid_scaler(attn_out)

    # Attn Skip Connection -- Note that we skip connect here to the quantized residual
    x_quantized_residual = attn_out + x_quantized_residual

    if block.quantization_dict["quantize_asic_prenorm"]:
        num_bits = block.quantization_dict["quantize_asic_bits"]
        quant_method = block.quantization_dict["activations_quant_method"]
        x_quantized_residual = fake_quantize_act(block, "asic_mlp_prenorm", x_quantized_residual, num_bits, quant_method, iter_num)

    # MLP
    x_mlp_in = x_quantized_residual

    # MLP Pre-LN
    if block.use_pre_ln:
        x_mlp_in = block.pre_ln_mlp(x_mlp_in)

    # MLP Operation
    mlp_out = block.mlp(x_mlp_in, iter_num)

    # MLP Peri-LN
    if block.use_peri_ln:
        mlp_out = block.out_ln_mlp(mlp_out)

    # MLP Output Scaling
    if block.mlp_resid_scaler is not None:
        mlp_out = block.mlp_resid_scaler(mlp_out)

    chip_output = mlp_out + x_quantized_residual

    # Off-Chip: Merge Quantized Residual With Full Precision Residual
    # Note:
    # chip_output = x_quantized_residual_initial + mlp_out + attn_out
    # Therefore subtract initial before mergin
    x = (chip_output - x_quantized_residual_initial) + x

    # Off-Chip: MLP Post-LN
    if block.use_post_ln:
        x = block.post_ln_mlp(x)

    return x


block_forward_variations = {
    "parallel_mlp": parallel_mlp_forward,
    "attn_then_mlp": attn_then_mlp_forward,
    "edgellm_asic": edgellm_asic_forward,
}


# -----------------------
# Normalization helpers
# -----------------------

def _resolve_unit_norm_flags(self, config) -> None:
    """Populate per-unit norm flags from config with per-position defaults."""
    NORM_POSITIONS = ("pre", "post", "peri")
    BLOCK_UNITS    = ("attn", "mlp")

    for unit in BLOCK_UNITS:
        for pos in NORM_POSITIONS:
            # granular setting and value (value may be None)
            granular_key = f"use_{pos}_ln_{unit}"
            granular_val = getattr(config, granular_key, None)

            # general setting and value (always defined)
            general_key  = f"use_{pos}_ln"
            general_val  = getattr(config, general_key, False)

            # Override general setting to granular setting, if a granular setting specified
            setattr(self, granular_key, granular_val if (granular_val is not None) else general_val)


def _setup_norms_parallel(self, config, norm_cls) -> None:
    """Norm layout for the 'parallel_mlp' variation."""
    # Pre-LN
    if getattr(self, "use_pre_ln", False):
        self.pre_ln = norm_cls(config)

    # Peri-LN
    if getattr(self, "use_peri_ln_attn", False):
        self.peri_ln_attn = norm_cls(config)
    if getattr(self, "use_peri_ln_mlp", False):
        self.peri_ln_mlp = norm_cls(config)

    # Post-LN
    if getattr(self, "use_post_ln", False):
        self.post_ln = norm_cls(config)

def _setup_norms_sequential(self, config, norm_cls) -> None:
    """Norm layout for the 'attn_then_mlp' variation."""

    # Pre-Norm
    if getattr(self, "use_pre_ln_attn", False):
        self.pre_ln_attn = norm_cls(config)
    if getattr(self, "use_pre_ln_mlp", False):
        self.pre_ln_mlp = norm_cls(config)

    # Peri-LN
    if getattr(self, "use_peri_ln_attn", False):
        self.out_ln_attn = norm_cls(config)
    if getattr(self, "use_peri_ln_mlp", False):
        self.out_ln_mlp = norm_cls(config)

    # Post-LN
    if getattr(self, "use_post_ln_attn", False):
        self.post_ln_attn = norm_cls(config)
    if getattr(self, "use_post_ln_mlp", False):
        self.post_ln_mlp = norm_cls(config)


normalization_setup_variations = {
    "parallel_mlp": _setup_norms_parallel,
    "attn_then_mlp": _setup_norms_sequential,
    "edgellm_asic": _setup_norms_sequential,
}


# -----------------------
# Residual scaler helpers
# -----------------------

def _setup_resid_scalers_parallel(self, config) -> None:
    """Residual scalers for 'parallel_mlp' variation (per-branch)."""
    self.attn_resid_scaler = None
    self.mlp_resid_scaler  = None

    if getattr(config, "use_attn_resid_scaling", False):
        cls = learned_confidence_dictionary[config.attn_confidence_variant]
        self.attn_resid_scaler = cls(config, prefix="attn")

    if getattr(config, "use_mlp_resid_scaling", False):
        cls = learned_confidence_dictionary[config.mlp_confidence_variant]
        self.mlp_resid_scaler = cls(config, prefix="mlp")


def _setup_resid_scalers_sequential(self, config) -> None:
    """Residual scalers for 'attn_then_mlp' variation (per-branch)."""
    # Kept identical to parallel; separated for future flexibility.
    self.attn_resid_scaler = None
    self.mlp_resid_scaler  = None

    if getattr(config, "use_attn_resid_scaling", False):
        cls = learned_confidence_dictionary[config.attn_confidence_variant]
        self.attn_resid_scaler = cls(config, prefix="attn")

    if getattr(config, "use_mlp_resid_scaling", False):
        cls = learned_confidence_dictionary[config.mlp_confidence_variant]
        self.mlp_resid_scaler = cls(config, prefix="mlp")


resid_scaler_setup_variations = {
    "parallel_mlp": _setup_resid_scalers_parallel,
    "attn_then_mlp": _setup_resid_scalers_sequential,
    "edgellm_asic": _setup_resid_scalers_sequential,
}


class Block(nn.Module):
    """Transformer block supporting multiple normalization strategies."""

    def __init__(self, config, mlp=None, attn=None):
        super().__init__()

        # Choose norm class for attention/MLP blocks
        norm_cls = norm_dictionary[config.norm_variant_attn]

        # Resolve per-unit norm flags from config (pre/post/peri × attn/mlp)
        _resolve_unit_norm_flags(self, config)

        # Aggregate flags (if referenced elsewhere)
        self.use_pre_ln  = getattr(config, "use_pre_ln",  False)
        self.use_post_ln = getattr(config, "use_post_ln", False)
        self.use_peri_ln = getattr(config, "use_peri_ln", False)

        # Forward variation choice
        self.use_parallel_mlp = getattr(config, "use_parallel_mlp", False)
        self.use_edgellm_asic = getattr(config, "use_edgellm_asic", False)

        if self.use_parallel_mlp:
            variant = "parallel_mlp"
        elif self.use_edgellm_asic:
            variant = "edgellm_asic"
            # Special Quantization Setup
            self.quantization_dict = {}
            self.quantization_dict["quantize_asic_prenorm"] = config.quantize_asic_prenorm
            self.quantization_dict["quantize_asic_bits"] = config.quantize_asic_bits
            self.quantization_dict["activations_quant_method"] = config.activations_quant_method
            self.full_quant_iteration = config.full_quant_iteration
            self.eval_interval = config.eval_interval
            self.start_quant_level = config.start_quant_level
            self.quant_scheduler = config.quant_scheduler
        else:
            variant = "attn_then_mlp"

        # Set Block Forward Variant
        self.block_forward = block_forward_variations[variant]

        ## Instantiate norms for Block Forward Variant
        normalization_setup_variations[variant](self, config, norm_cls)

        ## Instantiate (Optional) learned residual scalers for Block Forward Variant
        resid_scaler_setup_variations[variant](self, config)

        ## Instantiate Block Forward Variant Submodules
        if attn is None:
            self.attn = attention_dictionary[config.attention_variant](config)
        else:
            self.attn = attn

        if mlp is None:
            self.mlp = get_mlp_instance(config)
        else:
            self.mlp = mlp

        # Gradient checkpointing
        self.use_gradient_checkpointing = getattr(config, "use_gradient_checkpointing", False)

    def forward(self, x: torch.Tensor, iter_num: int):
        if self.use_gradient_checkpointing and x.requires_grad:
            return checkpoint.checkpoint(self.block_forward, self, x, iter_num, use_reentrant=False)
        return self.block_forward(self, x, iter_num)

