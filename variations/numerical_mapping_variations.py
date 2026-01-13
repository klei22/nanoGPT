# variations/numerical_mapping_variations.py

import torch
import torch.nn as nn
from torch.nn import functional as F


class NumericalMLPEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.numerical_mlp_hidden_dim
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, config.n_embd),
        )

    def forward(self, x):
        return self.net(x)


class NumericalMLPOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.numerical_mlp_hidden_dim
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class NumericalLinearEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(1, config.n_embd)

    def forward(self, x):
        return self.proj(x)


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


numerical_embedding_dictionary = {
    "mlp": NumericalMLPEmbedding,
    "linear": NumericalLinearEmbedding,
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
