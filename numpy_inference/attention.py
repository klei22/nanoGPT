import numpy as np
from .layers import Linear, LayerNorm


class CausalSelfAttention:
    def __init__(self, n_embd: int, n_head: int, block_size: int,
                 q_weight, k_weight, v_weight, proj_weight,
                 q_bias=None, k_bias=None, v_bias=None, proj_bias=None,
                 quant_bits=None):
        self.n_head = n_head
        self.block_size = block_size
        head_dim = n_embd // n_head
        self.q_proj = Linear(q_weight, q_bias, quant_bits)
        self.k_proj = Linear(k_weight, k_bias, quant_bits)
        self.v_proj = Linear(v_weight, v_bias, quant_bits)
        self.proj = Linear(proj_weight, proj_bias, quant_bits)
        self.bias = np.tril(np.ones((block_size, block_size), dtype=np.float32))
        self.head_dim = head_dim

    def __call__(self, x: np.ndarray, cache=None):
        B, T, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            k = np.concatenate([cache['k'], k], axis=2)
            v = np.concatenate([cache['v'], v], axis=2)
        else:
            cache = {'k': k, 'v': v}

        att = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        mask = self.bias[:T, :T]
        att = np.where(mask == 0, -1e10, att)
        att = np.exp(att - np.max(att, axis=-1, keepdims=True))
        att = att / np.sum(att, axis=-1, keepdims=True)
        y = np.matmul(att, v)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        y = self.proj(y)
        return y, cache
