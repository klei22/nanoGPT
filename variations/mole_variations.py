# variations/mole_variations.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from variations.router_variations import router_dictionary
from variations.mlp_variations import get_mlp_instance

class LookupTable(nn.Module):
    """Simple lookup table storing pre-computed expert outputs."""
    def __init__(self, vocab_size, out_dim):
        super().__init__()
        self.register_buffer("table", torch.zeros(vocab_size, out_dim))

    def forward(self, input_ids):
        return self.table[input_ids]

class MoLELayer(nn.Module):
    """Mixture of Lookup Experts layer.

    During training this behaves similarly to MoE but all experts are
    activated and take embedding tokens as input. Before inference the
    experts can be reparameterized into a lookup table via `build_lut`.
    """
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.n_experts
        scheme = getattr(config, "mole_router_scheme", config.moe_router_scheme)
        self.router = router_dictionary[scheme](config)
        self.shared_expert = get_mlp_instance(config)
        self.routed_expert = nn.ModuleList(
            [get_mlp_instance(config) for _ in range(config.n_experts)]
        )
        self.expert_layernorm = nn.LayerNorm(config.n_embd)
        self.lut = None
        self.config = config
        self.is_mole = True

    def build_lut(self, embedding_weights):
        """Pre-compute expert outputs for each token.

        Args:
            embedding_weights (Tensor): embedding weight matrix of shape
                (vocab_size, n_embd).
        """
        with torch.no_grad():
            outputs = []
            emb = self.expert_layernorm(embedding_weights)
            for expert in self.routed_expert:
                out, _ = expert(emb)
                outputs.append(out.unsqueeze(1))
            lut = torch.cat(outputs, dim=1)
        self.lut = LookupTable(lut.size(0), lut.size(1) * lut.size(2))
        self.lut.table.copy_(lut.view(lut.size(0), -1))
        # free expert parameters
        for p in self.routed_expert.parameters():
            p.requires_grad = False
        self.routed_expert = None

    def forward(self, hidden_states, iter_num=None, mlp_res=None, embedding_states=None, input_ids=None):
        """Forward pass that mirrors ``OriginalMLP`` signature with extra MoLE args."""

        if self.lut is not None and input_ids is not None:
            lookup = self.lut(input_ids).to(hidden_states.device, non_blocking=True)
            lookup = lookup.view(*input_ids.shape, self.num_experts, self.config.n_embd)
            residual = hidden_states
            router_value, _ = self.router(hidden_states)
            hidden_states = self.post_attention_layernorm(hidden_states)
            shared_output, _ = self.shared_expert(hidden_states, iter_num)
            routed_output = (lookup * router_value.unsqueeze(-1)).sum(dim=-2)
            hidden_states = residual + shared_output + routed_output
            return hidden_states, mlp_res

        assert embedding_states is not None, "embedding_states required during training"

        residual = hidden_states
        router_value, _ = self.router(hidden_states)
        shared_output, _ = self.shared_expert(hidden_states, iter_num)
        emb = self.expert_layernorm(embedding_states)
        expert_outs = []
        for expert in self.routed_expert:
            out, _ = expert(emb, iter_num)
            expert_outs.append(out.unsqueeze(2))
        routed_output = torch.cat(expert_outs, dim=2)
        routed_output = (routed_output * router_value.unsqueeze(-1)).sum(dim=2)
        hidden_states = residual + shared_output + routed_output
        return hidden_states, mlp_res

