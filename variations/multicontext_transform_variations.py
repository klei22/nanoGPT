# variations/multicontext_transform_variations.py

import torch
import torch.nn as nn


def _build_random_orthonormal_matrix(dim, device=None, dtype=None):
    matrix = torch.randn(dim, dim, device=device, dtype=dtype)
    q, r = torch.linalg.qr(matrix)
    diag = torch.sign(torch.diag(r))
    q = q * diag
    return q


class IdentityMulticontextTransform(nn.Module):
    def forward(self, x):
        return x


class LearnedVectorAddition(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vector = nn.Parameter(torch.empty(1, config.n_embd))
        nn.init.normal_(self.vector, mean=0.0, std=0.02)

    def forward(self, x):
        vector = self.vector.to(device=x.device, dtype=x.dtype)
        return x + vector


class RandomOrthonormalTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        matrix = _build_random_orthonormal_matrix(config.n_embd)
        self.register_buffer("matrix", matrix)

    def forward(self, x):
        matrix = self.matrix.to(device=x.device, dtype=x.dtype)
        return torch.matmul(x, matrix)


class LearnedOrthonormalTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(config.n_embd, config.n_embd))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x):
        weight = self.weight.to(device=x.device, dtype=x.dtype)
        q, _ = torch.linalg.qr(weight)
        return torch.matmul(x, q)


multicontext_transform_dictionary = {
    "none": IdentityMulticontextTransform,
    "learned_vector_add": LearnedVectorAddition,
    "random_orthonormal": RandomOrthonormalTransform,
    "learned_orthonormal": LearnedOrthonormalTransform,
}


def get_multicontext_transform(config):
    variant = config.multicontext_transform_variant
    cls = multicontext_transform_dictionary.get(variant)
    if cls is None:
        raise ValueError(f"Unsupported multicontext transform variant: {variant}")
    return cls(config)
