"""Router variations for mixture-of-experts and mixture-of-attention."""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class TopKRouter(nn.Module):
    """ Conventional Softmax Top_k Gating network (router) NN for MoE layers """
    def __init__(self, config):
        super().__init__()
        self.top_k = config.moe_top_k
        self.moe_router_scheme = config.moe_router_scheme
        self.route_linear = nn.Linear(config.n_embd, config.n_experts)

    def forward(self, x):
        logits = self.route_linear(x)

        top_k_logits, indices = logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(logits, float('-inf'))

        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output= F.softmax(sparse_logits, dim=-1)

        return router_output, indices


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
        noise = torch.randn_like(logits)*F.softplus(noise_logits)

        top_k_noisy_logits = noise_logits + noise
        top_k_logits, indices = logits.topk(self.top_k, dim=1)

        zeros = torch.full_like(top_k_noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)

        router_output = F.softmax(sparse_logits, dim=-1)

        return router_output, indices


class MoATopKRouter(nn.Module):
    """Top-k router for MoA query heads."""

    def __init__(self, config):
        super().__init__()
        self.top_k = getattr(config, "moa_top_k", 0)
        self.route_linear = nn.Linear(config.n_embd, config.n_moa_head)

    def forward(self, x):
        logits = self.route_linear(x)
        if self.top_k <= 0 or self.top_k >= logits.size(-1):
            gate = F.softmax(logits, dim=-1)
            indices = torch.arange(logits.size(-1), device=x.device)
            return gate, indices
        top_k_logits, indices = logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        gate = F.softmax(sparse_logits, dim=-1)
        return gate, indices


class MoAThresholdRouter(nn.Module):
    """Sigmoid router that activates heads above a threshold."""

    def __init__(self, config):
        super().__init__()
        self.threshold = getattr(config, "moa_threshold", 0.0)
        self.route_linear = nn.Linear(config.n_embd, config.n_moa_head)

    def forward(self, x):
        logits = torch.sigmoid(self.route_linear(x))
        gate = (logits > self.threshold).float() * logits
        return gate, None

router_dictionary = {
    "softmax": TopKRouter,
    "noisy_top_k": NoisyTopKRouter,
    "moa_topk": MoATopKRouter,
    "moa_threshold": MoAThresholdRouter,
}
