"""Block-level layernorm ordering strategies."""

from typing import Callable, Tuple
import torch

# type alias for the forward function
BlockForward = Callable[["Block", torch.Tensor, int, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


def post_ln_forward(block, x, iter_num, mlp_res=None):
    """Standard Post-LN behavior."""
    if block.use_parallel_mlp:
        x = block.ln_1(x + block.attn(x, iter_num) + block.mlp(x, iter_num))
        return x, mlp_res
    else:
        x = block.ln_1(x + block.attn(x, iter_num))
        x = block.ln_2(x + block.mlp(x, iter_num))
        return x, mlp_res


def pre_ln_forward(block, x, iter_num, mlp_res=None):
    """Pre-LN arrangement."""
    if block.use_parallel_mlp:
        ln_1 = block.ln_1(x)
        mlp_out, mlp_res = block.mlp(ln_1, iter_num)
        x = x + block.attn(ln_1, iter_num) + mlp_out
        return x, mlp_res
    else:
        x = x + block.attn(block.ln_1(x), iter_num)
        mlp_out, mlp_res = block.mlp(block.ln_2(x), iter_num, mlp_res)
        x = x + mlp_out
        return x, mlp_res


def peri_ln_forward(block, x, iter_num, mlp_res=None):
    """Peri-LN places normalization around each sublayer."""
    if block.use_parallel_mlp:
        ln_1 = block.ln_1(x)
        attn_out = block.out_ln_attn(block.attn(ln_1, iter_num))
        mlp_out, mlp_res = block.mlp(ln_1, iter_num)
        mlp_out = block.out_ln_mlp(mlp_out)
        x = x + attn_out + mlp_out
        return x, mlp_res
    else:
        attn_out = block.out_ln_attn(block.attn(block.ln_1(x), iter_num))
        x = x + attn_out
        mlp_out, mlp_res = block.mlp(block.ln_2(x), iter_num, mlp_res)
        mlp_out = block.out_ln_mlp(mlp_out)
        x = x + mlp_out
        return x, mlp_res


block_forward_variations = {
    "post_ln": post_ln_forward,
    "pre_ln": pre_ln_forward,
    "peri_ln": peri_ln_forward,
}

