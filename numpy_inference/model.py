import pickle
from typing import List
import numpy as np

from .layers import LayerNorm
from .block import Block


class Transformer:
    def __init__(self, n_layer: int, n_head: int, n_embd: int, block_size: int,
                 vocab_size: int, weights: dict, quant_bits: int | None = None):
        self.wte = weights['transformer.wte.weight']
        self.wpe = weights['transformer.wpe.weight']
        self.blocks: List[Block] = []
        for i in range(n_layer):
            prefix = f'transformer.h.{i}.'
            block_weights = {
                'ln_1.weight': weights[prefix + 'ln_1.weight'],
                'attn.c_attn_q.weight': weights[prefix + 'attn.c_attn_q.weight'],
                'attn.c_attn_k.weight': weights[prefix + 'attn.c_attn_k.weight'],
                'attn.c_attn_v.weight': weights[prefix + 'attn.c_attn_v.weight'],
                'attn.c_proj.weight': weights[prefix + 'attn.c_proj.weight'],
                'ln_2.weight': weights[prefix + 'ln_2.weight'],
                'mlp.c_fc.weight': weights[prefix + 'mlp.c_fc.weight'],
                'mlp.c_proj.weight': weights[prefix + 'mlp.c_proj.weight'],
            }
            self.blocks.append(Block(n_embd, n_head, block_size, block_weights, quant_bits))
        self.ln_f = LayerNorm(weights['transformer.ln_f.gain'], None)
        self.vocab_size = vocab_size
        self.block_size = block_size

    def forward(self, idx: np.ndarray, cache=None):
        B, T = idx.shape
        x = self.wte[idx] + self.wpe[:T]
        new_cache = []
        for i, block in enumerate(self.blocks):
            block_cache = None if cache is None else cache[i]
            x, bc = block(x, block_cache)
            new_cache.append(bc)
        x = self.ln_f(x)
        return x, new_cache

    def generate(self, idx: np.ndarray, max_new_tokens: int):
        cache = None
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.shape[1] <= self.block_size else idx[:, -self.block_size:]
            logits, cache = self.forward(idx_cond, cache)
            next_token = np.argmax(logits[:, -1, :], axis=-1)
            idx = np.concatenate((idx, next_token[:, None]), axis=1)
        return idx


def load_weights(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)
