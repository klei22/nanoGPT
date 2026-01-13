# variations/numerical_mapping_variations.py

import torch.nn as nn


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


def get_numerical_output(config):
    variant = config.numerical_output_variant
    cls = numerical_output_dictionary.get(variant)
    if cls is None:
        raise ValueError(f"Unsupported numerical output variant: {variant}")
    return cls(config)
