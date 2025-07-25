# variations/mole_variations.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from variations.router_variations import router_dictionary
from variations.mlp_variations import get_mlp_instance

class MoLELayer(nn.Module):
    """Mixture of Lookup Experts layer."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.n_experts
        self.router = router_dictionary[config.moe_router_scheme](config)
        self.shared_expert = get_mlp_instance(config)
        self.experts = nn.ModuleList([get_mlp_instance(config) for _ in range(self.num_experts)])
        self.use_lut = getattr(config, "use_mole_lut", False)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.n_embd
        self.lut = None

    def precompute_lut(self, embedding_weight: torch.Tensor):
        """Pre-compute expert outputs for all tokens."""
        with torch.no_grad():
            device = embedding_weight.device
            lut = torch.zeros(self.vocab_size, self.num_experts, self.hidden_size, device=device)
            for i, expert in enumerate(self.experts):
                lut[:, i, :] = expert(embedding_weight)
            self.lut = lut.cpu()

    def forward(self, x, iter_num=None, mlp_res=None, embedding_tokens=None, input_ids=None):
        router_value = F.softmax(self.router(x), dim=-1)
        if self.use_lut and self.lut is not None and input_ids is not None:
            lookup = self.lut[input_ids].to(x.device)
            routed_output = (lookup * router_value.unsqueeze(-1)).sum(dim=2)
        else:
            assert embedding_tokens is not None, "embedding_tokens required for MoLE training"
            expert_outputs = [expert(embedding_tokens) for expert in self.experts]
            stacked = torch.stack(expert_outputs, dim=2)
            routed_output = (stacked * router_value.unsqueeze(-1)).sum(dim=2)
        shared_out, _ = self.shared_expert(x, iter_num, mlp_res)
        out = shared_out + routed_output
        return out, mlp_res
