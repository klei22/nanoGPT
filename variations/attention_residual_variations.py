"""Depth-wise attention residuals.

This module implements Full Attention Residuals: each Transformer sublayer gets
an input selected from the embedding and all earlier sublayer outputs.  Routing
is token-local (the softmax dimension is depth), so sequence mixing remains the
responsibility of the normal self-attention module.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class FullAttentionResidual(nn.Module):
    """Mix earlier sublayer outputs with zero-initialized pseudo-queries."""

    def __init__(
        self,
        n_destinations: int,
        n_embd: int,
        eps: float = 1e-6,
        weighting: nn.Module | None = None,
        use_qk_norm: bool = False,
        use_qk_norm_scale: bool = False,
    ):
        super().__init__()
        # Includes one destination for each attention/MLP and one for ln_f.
        self.queries = nn.Parameter(torch.zeros(n_destinations, n_embd))
        self.eps = eps
        self.weighting = weighting
        self.use_qk_norm = use_qk_norm
        self.use_qk_norm_scale = use_qk_norm_scale
        if self.use_qk_norm_scale:
            depth = max(n_destinations, 2)
            initial_scale = math.log2(depth * depth - depth)
            self.qk_norm_factor = nn.Parameter(torch.tensor(initial_scale))

    @torch.no_grad()
    def initialize_queries(self, variant: str, scale: float) -> None:
        """Initialize routing queries for non-softmax weighting functions."""
        initializers = {
            "zeros": lambda tensor: nn.init.zeros_(tensor),
            "ones": lambda tensor: nn.init.ones_(tensor),
            "constant": lambda tensor: nn.init.constant_(tensor, scale),
            "normal": lambda tensor: nn.init.normal_(tensor, std=scale),
            "uniform": lambda tensor: nn.init.uniform_(tensor, -scale, scale),
            "positive_uniform": lambda tensor: nn.init.uniform_(tensor, 0.0, scale),
            "xavier_normal": lambda tensor: nn.init.xavier_normal_(tensor, gain=scale),
            "xavier_uniform": lambda tensor: nn.init.xavier_uniform_(tensor, gain=scale),
        }
        if variant not in initializers:
            raise ValueError(f"unknown attention residual query initialization: {variant}")
        initializers[variant](self.queries)
        if variant == "ones":
            self.queries.mul_(scale)

    def forward(self, sources: list[torch.Tensor], destination: int) -> torch.Tensor:
        if not sources:
            raise ValueError("attention residuals require at least one source")
        values = torch.stack(sources, dim=0)  # depth, batch, time, channels
        query = self.queries[destination]
        if self.use_qk_norm:
            # Match the repository's token-attention QK norm implementation.
            query = query / (query.norm(dim=-1, keepdim=True) + self.eps)
            keys = values / (values.norm(dim=-1, keepdim=True) + self.eps)
        else:
            keys = F.rms_norm(values, (values.size(-1),), eps=self.eps)
        scores = torch.einsum("dbtc,c->dbt", keys, query)
        if self.use_qk_norm_scale:
            scores = scores * self.qk_norm_factor
        elif self.use_qk_norm:
            scores = scores / math.sqrt(values.size(-1))
        weights = scores.softmax(dim=0) if self.weighting is None else self.weighting(scores)
        return torch.einsum("dbt,dbtc->btc", weights, values)
