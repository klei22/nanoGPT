"""Fused ReLU2Max attention built on PyTorch FlexAttention.

FlexAttention keeps the QK matmul, score transform, causal mask, and PV matmul
inside one generated kernel.  In particular, it does not materialize the
quadratic attention matrix that the eager ReLU2Max implementation creates.
"""

import importlib
import importlib.util

import torch


def flex_relu2max_available() -> bool:
    """Return whether this PyTorch build exposes the required kernel API."""
    return importlib.util.find_spec("torch.nn.attention.flex_attention") is not None


def make_relu2max_score_mod(divisor: float, sequence_length: int | None = None):
    """Create the pointwise score transform used by ReLU2Max attention."""
    scale = float(divisor)
    if scale <= 0:
        raise ValueError("relu2max_divisor must be greater than zero")
    if sequence_length is not None:
        scale *= sequence_length

    def relu2max(score, _batch, _head, _query, _key):
        return torch.relu(score).square() / scale

    return relu2max


class FusedReLU2MaxAttention:
    """Shape-aware facade around the compiled FlexAttention kernel."""

    def __init__(self, divisor: float, div_by_seq_len: bool):
        flex_module = importlib.import_module("torch.nn.attention.flex_attention")
        self._create_block_mask = flex_module.create_block_mask
        self.divisor = divisor
        self.div_by_seq_len = div_by_seq_len
        self._causal_masks = {}
        self._score_mods = {}
        # Compiling the FlexAttention higher-order op is what lowers it to a
        # tiled CUDA kernel instead of executing its eager reference path.
        self._kernel = torch.compile(flex_module.flex_attention, dynamic=False)

    def _causal_mask(self, length: int, device: torch.device):
        key = (length, str(device))
        if key not in self._causal_masks:
            def causal(_batch, _head, query, key_value):
                return query >= key_value

            self._causal_masks[key] = self._create_block_mask(
                causal, B=None, H=None, Q_LEN=length, KV_LEN=length,
                device=device,
            )
        return self._causal_masks[key]

    def __call__(self, query, key, value):
        length = query.size(-2)
        if length not in self._score_mods:
            self._score_mods[length] = make_relu2max_score_mod(
                self.divisor, length if self.div_by_seq_len else None
            )
        return self._kernel(
            query, key, value,
            score_mod=self._score_mods[length],
            block_mask=self._causal_mask(length, query.device),
        )
