# variations/router_variations.py
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import NamedTuple


class RouterOutput(NamedTuple):
    """Tuple-compatible router result with diagnostics for scientific logging."""
    probabilities: torch.Tensor
    indices: torch.Tensor
    logits: torch.Tensor
    selected_weights: torch.Tensor
    margins: torch.Tensor
    load: torch.Tensor


def _router_output(logits, top_k):
    selected_logits, indices = logits.topk(top_k, dim=-1)
    sparse_logits = torch.full_like(logits, float('-inf')).scatter(-1, indices, selected_logits)
    probabilities = F.softmax(sparse_logits, dim=-1)
    selected_weights = probabilities.gather(-1, indices)
    margin_k = min(top_k + 1, logits.shape[-1])
    margin_logits = logits.topk(margin_k, dim=-1).values
    margins = margin_logits[..., 0] - margin_logits[..., -1]
    # ``indices`` is [..., top_k]. After one-hot encoding it becomes
    # [..., top_k, n_experts], so reduce every original indices dimension,
    # including the top-k selection axis, and retain only n_experts.
    load = F.one_hot(indices, logits.shape[-1]).float().sum(tuple(range(indices.ndim)))
    return RouterOutput(probabilities, indices, logits, selected_weights, margins, load)

class TopKRouter(nn.Module):
    """ Conventional Softmax Top_k Gating network (router) NN for MoE layers """
    def __init__(self, config):
        super().__init__()
        self.top_k = config.moe_top_k
        self.moe_router_scheme = config.moe_router_scheme
        self.route_linear = nn.Linear(config.n_embd, config.n_experts)

    def forward(self, x):
        logits = self.route_linear(x)

        result = _router_output(logits, self.top_k)
        # Preserve the historical two-value API for existing MoE callers.
        return result.probabilities, result.indices

    def route_with_diagnostics(self, x):
        return _router_output(self.route_linear(x), self.top_k)


class NoisyTopKRouter(nn.Module):
    """ Noisy Top_k Gating network (router) NN for MoE layers """
    def __init__(self, config):
        super().__init__()
        self.top_k = config.moe_top_k
        self.moe_router_scheme = config.moe_router_scheme
        self.route_linear = nn.Linear(config.n_embd, config.n_experts)
        self.noise_linear = nn.Linear(config.n_embd, config.n_experts)

    def forward(self, x):
        logits = self.route_linear(x)

        noise_logits = self.noise_linear(x)
        routed_logits = logits + torch.randn_like(logits) * F.softplus(noise_logits)
        result = _router_output(routed_logits, self.top_k)
        return result.probabilities, result.indices

    def route_with_diagnostics(self, x):
        logits = self.route_linear(x)
        scale = F.softplus(self.noise_linear(x))
        return _router_output(logits + torch.randn_like(logits) * scale, self.top_k)

router_dictionary = {
    "softmax": TopKRouter,
    "noisy_top_k": NoisyTopKRouter,
}
