# variations/numerical_mapping_variations.py

import torch.nn as nn

from variations.activation_variations import activation_dictionary


def _build_activation(config):
    return activation_dictionary[config.numerical_mapping_activation](config=config)


class NumericalMLPMapping(nn.Module):
    def __init__(self, config, in_dim, out_dim):
        super().__init__()
        hidden_dim = config.numerical_mlp_hidden_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            _build_activation(config),
            nn.Linear(hidden_dim, hidden_dim),
            _build_activation(config),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class NumericalShallowMLPMapping(nn.Module):
    def __init__(self, config, in_dim, out_dim):
        super().__init__()
        hidden_dim = config.numerical_mlp_hidden_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            _build_activation(config),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class NumericalLinearMapping(nn.Module):
    def __init__(self, config, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


numerical_mapping_dictionary = {
    "mlp": NumericalMLPMapping,
    "shallow_mlp": NumericalShallowMLPMapping,
    "linear": NumericalLinearMapping,
}


def get_numerical_mapping(variant, config, in_dim, out_dim):
    mapping_cls = numerical_mapping_dictionary.get(variant)
    if mapping_cls is None:
        raise ValueError(f"Unsupported numerical mapping variant: {variant}")
    return mapping_cls(config, in_dim, out_dim)
