import torch
import torch.nn as nn
import torch.nn.functional as F

from variations.mlp_variations import get_mlp_instance
from variations.router_variations import router_dictionary
from variations.norm_variations import norm_dictionary

class MoLELayer(nn.Module):
    """Mixture of Lookup Experts layer.
    During training this behaves similar to an MoE layer but the experts take
    embedding tokens as input. At inference time the experts can be
    reparameterized as lookup tables (LUTs) and `use_lut` can be enabled.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.n_experts

        # Router used in both training and inference
        self.router = nn.Linear(config.n_embd, self.num_experts, bias=False)

        # Shared expert operates on hidden states like a standard MLP
        self.shared_expert = get_mlp_instance(config)

        # Routed experts. During inference their outputs are replaced by a LUT
        self.routed_experts = nn.ModuleList([
            get_mlp_instance(config) for _ in range(self.num_experts)
        ])

        self.expert_norm = norm_dictionary[config.norm_variant_attn](config)

        self.use_lut = False
        self.register_buffer("lut", None)

        # flag so Block knows to pass embedding tokens and input ids
        self.requires_embed_tokens = True

    def build_lut(self, embedding_table):
        """Pre-compute expert outputs for each vocabulary id."""
        with torch.no_grad():
            emb = self.expert_norm(embedding_table)
            outputs = []
            for expert in self.routed_experts:
                out, _ = expert(emb)
                outputs.append(out)
            lut = torch.stack(outputs, dim=1)  # (V, num_experts, hidden)
            self.lut = lut
            self.use_lut = True

    def forward(self, x, iter_num=None, mlp_res=None, embed_tokens=None, input_ids=None):
        gate = F.softmax(self.router(x), dim=-1)
        shared_out, mlp_res = self.shared_expert(x, iter_num, mlp_res)

        if self.use_lut:
            assert input_ids is not None and self.lut is not None
            b, t = input_ids.shape
            lut_vals = self.lut[input_ids.view(-1)]  # (b*t, num_experts, hidden)
            lut_vals = lut_vals.view(b, t, self.num_experts, -1)
            routed = torch.sum(lut_vals * gate.unsqueeze(-1), dim=2)
            return shared_out + routed, mlp_res
        else:
            assert embed_tokens is not None
            emb = self.expert_norm(embed_tokens)
            expert_outs = []
            for expert in self.routed_experts:
                out, _ = expert(emb, iter_num)
                expert_outs.append(out)
            stacked = torch.stack(expert_outs, dim=2)  # (b,t,num_experts,hidden)
            routed = torch.sum(stacked * gate.unsqueeze(-1), dim=2)
            return shared_out + routed, mlp_res
