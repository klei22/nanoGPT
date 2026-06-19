"""Block definitions and forward variations."""
from __future__ import annotations
from typing import Callable
from functools import partial
import copy
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
# Residual combination helpers
# -----------------------

def slerp(a: torch.Tensor, b: torch.Tensor, alpha: float, eps: float) -> torch.Tensor:
    """Spherical linear interpolation between tensors ``a`` and ``b``."""
    a_norm = a / a.norm(dim=-1, keepdim=True)
    b_norm = b / b.norm(dim=-1, keepdim=True)
    dot = (a_norm * b_norm).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    close = so.abs() < eps
    interp = (
        torch.sin((1 - alpha) * omega) / so * a
        + torch.sin(alpha * omega) / so * b
    )
    lerp = (1 - alpha) * a + alpha * b
    return torch.where(close, lerp, interp)


def add_residual(x: torch.Tensor, out: torch.Tensor, alpha: torch.Tensor, eps: float) -> torch.Tensor:
    return x + out


def rezero_residual(x: torch.Tensor, out: torch.Tensor, alpha: torch.Tensor, eps: float) -> torch.Tensor:
    return x + alpha * out


def lerp_residual(x: torch.Tensor, out: torch.Tensor, alpha: torch.Tensor, eps: float) -> torch.Tensor:
    return (1 - alpha) * x + alpha * (x + out)


def slerp_residual(x: torch.Tensor, out: torch.Tensor, alpha: torch.Tensor, eps: float) -> torch.Tensor:
    return slerp(x, x + out, alpha, eps)


residual_combine_dict = {
    "add": add_residual,
    "rezero": rezero_residual,
    "lerp": lerp_residual,
    "slerp": slerp_residual,
}


def make_alpha_fn(mode: str, init: float, param=None, vec=None):
    if mode == "fixed":
        return lambda _out: init
    if mode == "learned":
        return lambda _out: param
    if mode == "rezero":
        return lambda _out: param
    if mode == "dot":
        return lambda out: init * (param + (out * vec).sum(dim=-1, keepdim=True))
    raise ValueError(f"unknown alpha mode {mode}")

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
    if block.use_peri_ln_attn:
        attn_out = block.peri_ln_attn(attn_out)
    if block.use_peri_ln_mlp:
        mlp_out = block.peri_ln_mlp(mlp_out)

    # MLP and Attn Output Scaling
    if block.attn_resid_scaler is not None:
        attn_out = block.attn_resid_scaler(attn_out)
    if block.mlp_resid_scaler is not None:
        mlp_out = block.mlp_resid_scaler(mlp_out)

    # Skip Connection
    combined = attn_out + mlp_out
    x = block._combine_resid("attn", x, combined)

    # Post-LN
    if block.use_post_ln:
        x = block.post_ln(x)

    return x


def attn_then_mlp_forward(block, x: torch.Tensor, iter_num: int) -> torch.Tensor:
    """Attention followed by MLP."""

    # Make sure not to override skip connection
    x_attn_in = x

    # Attn Pre-LN
    if block.use_pre_ln_attn:
        x_attn_in = block.pre_ln_attn(x_attn_in)

    # Attn Operation
    attn_out = block.attn(x_attn_in, iter_num)

    # Attn Peri-LN
    if block.use_peri_ln_attn:
        attn_out = block.peri_ln_attn(attn_out)

    # Attn Output Scaling
    if block.attn_resid_scaler is not None:
        attn_out = block.attn_resid_scaler(attn_out)

    # Attn Skip Connection
    x = block._combine_resid("attn", x, attn_out)

    # Attn Post-LN
    if block.use_post_ln_attn:
        x = block.post_ln_attn(x)

    # Make sure not to override skip connection
    x_mlp_in = x

    # MLP Pre-LN
    if block.use_pre_ln_mlp:
        x_mlp_in = block.pre_ln_mlp(x_mlp_in)

    # MLP Operation
    mlp_out = block.mlp(x_mlp_in, iter_num)

    # MLP Peri-LN
    if block.use_peri_ln_mlp:
        mlp_out = block.peri_ln_mlp(mlp_out)

    # MLP Output Scaling
    if block.mlp_resid_scaler is not None:
        mlp_out = block.mlp_resid_scaler(mlp_out)

    # MLP Skip Connection
    x = block._combine_resid("mlp", x, mlp_out)

    # MLP Post-LN
    if block.use_post_ln_mlp:
        x = block.post_ln_mlp(x)

    return x


def edgellm_asic_forward(block, x: torch.Tensor, iter_num: int) -> torch.Tensor:
    """EdgeLLM ASIC forward: Attention followed by MLP with skip connection accumulation between blocks."""

    # Separate Full Precision Residual 'x' from 'x_quantized_residual'
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
    if block.use_pre_ln_attn:
        if block.use_flash_norm:
            print("Warning: FlashNorm is used with pre_ln, causing double normalization.")
        x_attn_in = block.pre_ln_attn(x_attn_in)

    # Attn Operation
    attn_out = block.attn(x_attn_in, iter_num)

    # Attn Peri-LN
    if block.use_peri_ln_attn:
        attn_out = block.peri_ln_attn(attn_out)

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
    if block.use_pre_ln_mlp:
        if block.use_flash_norm:
            print("Warning: FlashNorm is used with pre_ln, causing double normalization.")
        x_mlp_in = block.pre_ln_mlp(x_mlp_in)

    # MLP Operation
    mlp_out = block.mlp(x_mlp_in, iter_num)

    # MLP Peri-LN
    if block.use_peri_ln_mlp:
        mlp_out = block.peri_ln_mlp(mlp_out)

    # MLP Output Scaling
    if block.mlp_resid_scaler is not None:
        mlp_out = block.mlp_resid_scaler(mlp_out)

    chip_output = mlp_out + x_quantized_residual

    # Off-Chip: Merge Quantized Residual With Full Precision Residual
    # Note:
    # chip_output = x_quantized_residual_initial + mlp_out + attn_out
    # Therefore subtract initial before merging
    # x = (chip_output - x_quantized_residual_initial) + x
    adj_chip_output = chip_output - x_quantized_residual_initial
    x = block._combine_resid("mlp", x, adj_chip_output)

    if block.quantization_dict["quantize_asic_offchip_residual"]:
        num_bits = block.quantization_dict["quantize_asic_bits"]
        quant_method = block.quantization_dict["activations_quant_method"]
        x = fake_quantize_act(block, "asic_offchip_residual", x, num_bits, quant_method, iter_num)

    # Off-Chip: MLP Post-LN
    if block.use_post_ln_mlp:
        x = block.post_ln_mlp(x)

    return x


def sequential_ops_forward(block, x: torch.Tensor, iter_num: int) -> torch.Tensor:
    """Forward pass over a configurable op sequence (attn/mlp)."""

    residual_anchor = x
    chain_x = x

    for op_kind, op_module in block.op_sequence_modules:
        op_input = chain_x

        # Pre-LN
        if op_kind == "attn" and block.use_pre_ln_attn:
            op_input = block.pre_ln_attn(op_input)
        elif op_kind == "mlp" and block.use_pre_ln_mlp:
            op_input = block.pre_ln_mlp(op_input)

        op_out = op_module(op_input, iter_num)

        # Peri-LN
        if op_kind == "attn" and block.use_peri_ln_attn:
            op_out = block.peri_ln_attn(op_out)
        elif op_kind == "mlp" and block.use_peri_ln_mlp:
            op_out = block.peri_ln_mlp(op_out)

        # Optional learned residual scaling
        if op_kind == "attn" and block.attn_resid_scaler is not None:
            op_out = block.attn_resid_scaler(op_out)
        elif op_kind == "mlp" and block.mlp_resid_scaler is not None:
            op_out = block.mlp_resid_scaler(op_out)

        if block.sequence_intermediate_skip_connections:
            chain_x = block._combine_resid(op_kind, chain_x, op_out)

            # Post-LN after each sub-operation residual, if enabled
            if op_kind == "attn" and block.use_post_ln_attn:
                chain_x = block.post_ln_attn(chain_x)
            elif op_kind == "mlp" and block.use_post_ln_mlp:
                chain_x = block.post_ln_mlp(chain_x)
        else:
            # Pure sequential composition (no intermediate residual adds)
            chain_x = op_out

    if not block.sequence_intermediate_skip_connections and block.sequence_final_skip_connection:
        final_kind = block.sequence_kinds[-1]
        chain_x = block._combine_resid(final_kind, residual_anchor, chain_x)

        if final_kind == "attn" and block.use_post_ln_attn:
            chain_x = block.post_ln_attn(chain_x)
        elif final_kind == "mlp" and block.use_post_ln_mlp:
            chain_x = block.post_ln_mlp(chain_x)

    return chain_x


block_forward_variations = {
    "parallel_mlp": parallel_mlp_forward,
    "attn_then_mlp": attn_then_mlp_forward,
    "edgellm_asic": edgellm_asic_forward,
    "sequential_ops": sequential_ops_forward,
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
        self.peri_ln_attn = norm_cls(config)
    if getattr(self, "use_peri_ln_mlp", False):
        self.peri_ln_mlp = norm_cls(config)

    # Post-LN
    if getattr(self, "use_post_ln_attn", False):
        self.post_ln_attn = norm_cls(config)
    if getattr(self, "use_post_ln_mlp", False):
        self.post_ln_mlp = norm_cls(config)


normalization_setup_variations = {
    "parallel_mlp": _setup_norms_parallel,
    "attn_then_mlp": _setup_norms_sequential,
    "edgellm_asic": _setup_norms_sequential,
    "sequential_ops": _setup_norms_sequential,
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
    "sequential_ops": _setup_resid_scalers_sequential,
}


def _normalize_sequence_token(token: str) -> str:
    norm = token.strip().lower()
    if norm in {"attn", "attention"}:
        return "attn"
    if norm in {"mlp"}:
        return "mlp"
    raise ValueError(f"Unsupported block operation token '{token}'. Supported: attn, mlp")


def _resolve_block_sequence(config) -> list[str]:
    explicit = getattr(config, "block_operation_sequence", None)
    if explicit:
        return [_normalize_sequence_token(tok) for tok in explicit]

    attn_repeats = max(1, getattr(config, "block_attn_repeat", 1))
    mlp_repeats = max(1, getattr(config, "block_mlp_repeat", 1))
    return ["attn"] * attn_repeats + ["mlp"] * mlp_repeats


def _make_mlp_config_for_sequence(config, sequence_kinds: list[str]):
    """Split MLP width across repeated MLP ops to preserve base MLP parameter budget."""
    if not getattr(config, "match_sequential_mlp_param_budget", False):
        return config

    mlp_count = sum(1 for kind in sequence_kinds if kind == "mlp")
    if mlp_count <= 1:
        return config

    seq_config = copy.copy(config)
    base_mlp_size = config.mlp_size if config.mlp_size is not None else config.mlp_expansion_factor * config.n_embd
    seq_config.mlp_size = max(1, base_mlp_size // mlp_count)
    return seq_config


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
        self.use_sequential_ops = getattr(config, "use_sequential_ops", False)

        self.use_flash_norm = getattr(config, "use_flash_norm", False)

        if self.use_parallel_mlp:
            variant = "parallel_mlp"
        elif self.use_edgellm_asic:
            variant = "edgellm_asic"
            # Special Quantization Setup
            self.quantization_dict = {}
            self.quantization_dict["quantize_asic_prenorm"] = config.quantize_asic_prenorm
            self.quantization_dict["quantize_asic_offchip_residual"] = config.quantize_asic_offchip_residual
            self.quantization_dict["quantize_asic_bits"] = config.quantize_asic_bits
            self.quantization_dict["activations_quant_method"] = config.activations_quant_method
            self.full_quant_iteration = config.full_quant_iteration
            self.eval_interval = config.eval_interval
            self.start_quant_level = config.start_quant_level
            self.quant_scheduler = config.quant_scheduler
        elif self.use_sequential_ops:
            variant = "sequential_ops"
            self.sequence_kinds = _resolve_block_sequence(config)
            self.sequence_intermediate_skip_connections = getattr(config, "sequence_intermediate_skip_connections", True)
            self.sequence_final_skip_connection = getattr(config, "sequence_final_skip_connection", True)
        else:
            variant = "attn_then_mlp"

        # Set Block Forward Variant
        self.block_forward = partial(block_forward_variations[variant], self)

        ## Instantiate norms for Block Forward Variant
        normalization_setup_variations[variant](self, config, norm_cls)

        ## Instantiate (Optional) learned residual scalers for Block Forward Variant
        resid_scaler_setup_variations[variant](self, config)

        ## Instantiate Block Forward Variant Submodules
        if variant == "sequential_ops":
            seq_mlp_config = _make_mlp_config_for_sequence(config, self.sequence_kinds)

            self.attn_layers = nn.ModuleList()
            self.mlp_layers = nn.ModuleList()
            self.op_sequence_modules = []

            for kind in self.sequence_kinds:
                if kind == "attn":
                    module = attention_dictionary[config.attention_variant](config)
                    self.attn_layers.append(module)
                    self.op_sequence_modules.append((kind, module))
                else:
                    module = get_mlp_instance(seq_mlp_config)
                    self.mlp_layers.append(module)
                    self.op_sequence_modules.append((kind, module))
        else:
            if attn is None:
                self.attn = attention_dictionary[config.attention_variant](config)
            else:
                self.attn = attn

            if mlp is None:
                self.mlp = get_mlp_instance(config)
            else:
                self.mlp = mlp

        self.attn_resid_type = getattr(config, "attn_residual_combination", "add")
        self.mlp_resid_type = getattr(config, "mlp_residual_combination", "add")
        self.residual_slerp_eps = getattr(config, "residual_slerp_eps", 0.0)

        self.attn_alpha = getattr(config, "attn_residual_alpha", 0.05)
        self.mlp_alpha = getattr(config, "mlp_residual_alpha", 0.05)
        self.attn_alpha_mode = getattr(config, "attn_residual_alpha_type", "fixed")
        self.mlp_alpha_mode = getattr(config, "mlp_residual_alpha_type", "fixed")

        if self.attn_alpha_mode == "rezero":
            self.attn_alpha = 0.0
            self.attn_alpha_param = nn.Parameter(torch.tensor(0.0))
        elif self.attn_alpha_mode == "learned":
            self.attn_alpha_param = nn.Parameter(torch.tensor(self.attn_alpha))
        elif self.attn_alpha_mode == "dot":
            self.attn_alpha_vec = nn.Parameter(torch.zeros(config.n_embd))
            self.attn_alpha_param = nn.Parameter(torch.tensor(self.attn_alpha))
        if self.mlp_alpha_mode == "rezero":
            self.mlp_alpha = 0.0
            self.mlp_alpha_param = nn.Parameter(torch.tensor(0.0))
        elif self.mlp_alpha_mode == "learned":
            self.mlp_alpha_param = nn.Parameter(torch.tensor(self.mlp_alpha))
        elif self.mlp_alpha_mode == "dot":
            self.mlp_alpha_vec = nn.Parameter(torch.zeros(config.n_embd))
            self.mlp_alpha_param = nn.Parameter(torch.tensor(self.mlp_alpha))

        self.alpha_fns = {
            "attn": make_alpha_fn(self.attn_alpha_mode, self.attn_alpha,
                                  getattr(self, "attn_alpha_param", None),
                                  getattr(self, "attn_alpha_vec", None)),
            "mlp": make_alpha_fn(self.mlp_alpha_mode, self.mlp_alpha,
                                 getattr(self, "mlp_alpha_param", None),
                                 getattr(self, "mlp_alpha_vec", None)),
        }

        self.resid_fns = {
            "attn": residual_combine_dict[self.attn_resid_type],
            "mlp": residual_combine_dict[self.mlp_resid_type],
        }

        # Gradient checkpointing
        self.use_gradient_checkpointing = getattr(config, "use_gradient_checkpointing", False)

    def forward(self, x: torch.Tensor, iter_num: int):
        if self.use_gradient_checkpointing and x.requires_grad:
            return checkpoint.checkpoint(self.block_forward, x, iter_num, use_reentrant=False)
        return self.block_forward(x, iter_num)

    def _combine_resid(self, kind: str, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Helper method to streamline forward block skip connections"""
        alpha = self.alpha_fns[kind](out)
        return self.resid_fns[kind](x, out, alpha, self.residual_slerp_eps)
