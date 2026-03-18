# variations/numerical_mapping_variations.py

import copy

import torch
import torch.nn as nn
from torch.nn import functional as F

from variations.activation_variations import activation_dictionary
from variations.norm_variations import norm_dictionary


def _get_numerical_mlp_hidden_dims(config):
    if config.numerical_mlp_hidden_dims:
        return list(config.numerical_mlp_hidden_dims)
    if config.numerical_mlp_num_layers < 0:
        raise ValueError("numerical_mlp_num_layers must be non-negative")
    return [config.numerical_mlp_hidden_dim] * config.numerical_mlp_num_layers


def _build_numerical_mlp(config, input_dim, output_dim):
    hidden_dims = _get_numerical_mlp_hidden_dims(config)
    activation_cls = activation_dictionary[config.numerical_mlp_activation_variant]

    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation_cls(config=config))
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


def _build_channel_norm(config):
    variant = getattr(config, "norm_channel_variant", None)
    if variant is None:
        return None

    if variant not in norm_dictionary:
        raise ValueError(f"Unsupported norm_channel_variant: {variant}")

    norm_config = copy.deepcopy(config)
    for attr in ("radius", "scale", "gain", "radius_learning"):
        value = getattr(config, f"norm_channel_{attr}", None)
        if value is not None:
            setattr(norm_config, f"hsnorm_{attr}", value)
    return norm_dictionary[variant](norm_config)


class NumericalMLPEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = _build_numerical_mlp(config, 1, config.n_embd)
        self.channel_norm = _build_channel_norm(config)

    def forward(self, x):
        out = self.net(x)
        if self.channel_norm is not None:
            out = self.channel_norm(out)
        return out


class NumericalMLPOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = _build_numerical_mlp(config, config.n_embd, 1)

    def forward(self, x):
        return self.net(x)


class NumericalLinearEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(1, config.n_embd)
        self.channel_norm = _build_channel_norm(config)

    def forward(self, x):
        out = self.proj(x)
        if self.channel_norm is not None:
            out = self.channel_norm(out)
        return out


class NumericalCayleyEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.skew_param = nn.Parameter(torch.empty(self.n_embd, self.n_embd))
        self.vector = nn.Parameter(torch.empty(1, self.n_embd))
        self.channel_norm = _build_channel_norm(config)
        nn.init.normal_(self.skew_param, mean=0.0, std=0.02)
        nn.init.normal_(self.vector, mean=0.0, std=0.02)

    def forward(self, x):
        skew = self.skew_param - self.skew_param.t()
        scaled = x.unsqueeze(-1) * skew
        eye = torch.eye(self.n_embd, device=x.device, dtype=x.dtype)
        eye = eye.view(1, 1, self.n_embd, self.n_embd)
        q = torch.linalg.solve(eye - scaled, eye + scaled)
        vector = self.vector.to(device=x.device, dtype=x.dtype)
        out = torch.matmul(vector, q).squeeze(-2)
        if self.channel_norm is not None:
            out = self.channel_norm(out)
        return out


class NumericalLinearOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(config.n_embd, 1)

    def forward(self, x):
        return self.proj(x)


class NumericalLinearOutputTied(nn.Module):
    def __init__(self, embedding_module, bias=True):
        super().__init__()
        self.embedding_module = embedding_module
        if bias:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        weight = self.embedding_module.proj.weight
        return F.linear(x, weight.t(), self.bias)


class NumericalLearnedVectorScaledArcEmbedding(nn.Module):
    """Learned base vector rotated along a great-circle arc scaled by input.

    For each channel we learn:
      - v        : base vector on the embedding hypersphere  (1, n_embd)
      - d        : arc direction vector                       (1, n_embd)
      - theta_scale : scalar controlling the angular sweep

    Forward:
      d_perp = normalise(d - proj_v(d))          # orthogonal component
      out    = cos(x * theta_scale) * v  +  sin(x * theta_scale) * d_perp * ||v||
    """

    def __init__(self, config):
        super().__init__()
        self.vector = nn.Parameter(torch.empty(1, config.n_embd))
        self.arc_dir = nn.Parameter(torch.empty(1, config.n_embd))
        self.theta_scale = nn.Parameter(torch.ones(1))
        self.channel_norm = _build_channel_norm(config)
        nn.init.normal_(self.vector, mean=0.0, std=0.02)
        nn.init.normal_(self.arc_dir, mean=0.0, std=0.02)

    def forward(self, x):
        v = self.vector.to(device=x.device, dtype=x.dtype)
        d = self.arc_dir.to(device=x.device, dtype=x.dtype)
        ts = self.theta_scale.to(device=x.device, dtype=x.dtype)

        # Gram-Schmidt: remove projection of d onto v to get orthogonal component
        v_norm_sq = (v * v).sum(dim=-1, keepdim=True).clamp(min=1e-8)
        proj = (d * v).sum(dim=-1, keepdim=True) / v_norm_sq * v
        d_perp = d - proj
        d_perp_norm = d_perp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        d_perp = d_perp / d_perp_norm

        # Scale d_perp to match ||v|| so the arc stays on the same hypersphere
        v_norm = v.norm(dim=-1, keepdim=True)

        angle = x * ts  # (batch, seq, 1) * (1,) -> (batch, seq, 1)
        out = torch.cos(angle) * v + torch.sin(angle) * d_perp * v_norm

        if self.channel_norm is not None:
            out = self.channel_norm(out)
        return out


class NumericalScaledVectorEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vector = nn.Parameter(torch.empty(1, config.n_embd))
        self.channel_norm = _build_channel_norm(config)
        nn.init.normal_(self.vector, mean=0.0, std=0.02)

    def forward(self, x):
        vector = self.vector.to(device=x.device, dtype=x.dtype)
        out = x * vector
        if self.channel_norm is not None:
            out = self.channel_norm(out)
        return out


numerical_embedding_dictionary = {
    "mlp": NumericalMLPEmbedding,
    "linear": NumericalLinearEmbedding,
    "cayley": NumericalCayleyEmbedding,
    "scaled_vector": NumericalScaledVectorEmbedding,
    "learned_vector_scaled_arc": NumericalLearnedVectorScaledArcEmbedding,
}

numerical_output_dictionary = {
    "mlp": NumericalMLPOutput,
    "linear": NumericalLinearOutput,
}


def get_numerical_embedding(config):
    variant = config.numerical_embedding_variant
    cls = numerical_embedding_dictionary.get(variant)
    if cls is None:
        raise ValueError(f"Unsupported numerical embedding variant: {variant}")
    return cls(config)


def get_numerical_output(config, embedding_module=None):
    if (config.numerical_mapping_weight_tying
            and config.numerical_embedding_variant == "linear"
            and config.numerical_output_variant == "linear"
            and embedding_module is not None):
        return NumericalLinearOutputTied(embedding_module)
    variant = config.numerical_output_variant
    cls = numerical_output_dictionary.get(variant)
    if cls is None:
        raise ValueError(f"Unsupported numerical output variant: {variant}")
    return cls(config)
