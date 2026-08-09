"""Cayley-map orthogonal and semi-orthogonal layers."""

import torch
import torch.nn as nn
from torch.nn import functional as F


class CayleyLinear(nn.Module):
    """
    Orthogonal / semi-orthogonal Linear layer via a differentiable Cayley map.

    out_features <= in_features: W W.T ~= I  (orthonormal rows)
    out_features >= in_features: W.T W ~= I  (orthonormal columns)
    mode="exact": exact Cayley using torch.linalg.solve.
    mode="ns":    Newton-Schulz approximation to the Cayley inverse.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int | None = None,
        bias: bool = True,
        mode: str = "exact",
        ns_steps: int = 8,
        init_scale: float = 1e-3,
        max_skew_norm: float | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if out_features is None:
            out_features = in_features
        if mode not in {"exact", "ns"}:
            raise ValueError('mode must be "exact" or "ns"')
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.n = max(self.in_features, self.out_features)
        self.mode = mode
        self.ns_steps = int(ns_steps)
        self.max_skew_norm = max_skew_norm
        self.raw = nn.Parameter(init_scale * torch.randn(self.n, self.n, **factory_kwargs))
        self.bias = (
            nn.Parameter(torch.zeros(self.out_features, **factory_kwargs))
            if bias
            else None
        )
        self.register_buffer("_I", torch.eye(self.n, **factory_kwargs), persistent=False)

    @property
    def weight(self) -> torch.Tensor:
        return self.cayley_matrix()[: self.out_features, : self.in_features]

    def _skew(self) -> torch.Tensor:
        A = self.raw - self.raw.mT
        if self.max_skew_norm is not None:
            eps = torch.finfo(A.dtype).eps
            norm = torch.linalg.vector_norm(A).clamp_min(eps)
            cap = torch.as_tensor(self.max_skew_norm, dtype=A.dtype, device=A.device)
            A = A * (torch.tanh(norm / cap) * cap / norm)
        return A

    def _identity(self, A: torch.Tensor) -> torch.Tensor:
        return self._I.to(dtype=A.dtype, device=A.device)

    def _cayley_exact(self, A: torch.Tensor) -> torch.Tensor:
        I = self._identity(A)
        M = I - 0.5 * A
        N = I + 0.5 * A
        return torch.linalg.solve(M.mT, N.mT).mT

    def _cayley_ns_inverse(self, A: torch.Tensor) -> torch.Tensor:
        I = self._identity(A)
        M = I - 0.5 * A
        N = I + 0.5 * A
        eps = torch.finfo(A.dtype).eps
        alpha = 1.0 + 0.25 * torch.linalg.vector_norm(A).square()
        X = N / alpha.clamp_min(eps)
        for _ in range(self.ns_steps):
            X = X @ (2.0 * I - M @ X)
        return N @ X

    def cayley_matrix(self) -> torch.Tensor:
        A = self._skew()
        if self.mode == "exact":
            return self._cayley_exact(A)
        return self._cayley_ns_inverse(A)

    def orthogonality_error(self) -> torch.Tensor:
        W = self.weight
        if self.out_features <= self.in_features:
            G = W @ W.mT
            I = torch.eye(self.out_features, dtype=W.dtype, device=W.device)
        else:
            G = W.mT @ W
            I = torch.eye(self.in_features, dtype=W.dtype, device=W.device)
        return torch.linalg.matrix_norm(G - I)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
