import numpy as np
from .layers import LayerNorm
from .attention import CausalSelfAttention
from .mlp import MLP


class Block:
    def __init__(self, n_embd, n_head, block_size, weights, quant_bits=None):
        self.ln_1 = LayerNorm(weights['ln_1.weight'], None)
        self.attn = CausalSelfAttention(
            n_embd,
            n_head,
            block_size,
            weights['attn.c_attn_q.weight'],
            weights['attn.c_attn_k.weight'],
            weights['attn.c_attn_v.weight'],
            weights['attn.c_proj.weight'],
            quant_bits=quant_bits,
        )
        self.ln_2 = LayerNorm(weights['ln_2.weight'], None)
        self.mlp = MLP(
            weights['mlp.c_fc.weight'],
            weights['mlp.c_proj.weight'],
            quant_bits=quant_bits,
        )

    def __call__(self, x: np.ndarray, cache=None):
        attn_out, cache = self.attn(self.ln_1(x), cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, cache
