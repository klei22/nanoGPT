import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _hadamard(n: int, device=None) -> torch.Tensor:
    """Create a random Hadamard rotation matrix."""
    if n == 1:
        return torch.ones(1, 1, device=device)
    # Hadamard requires n power of two; fall back to QR if not
    if n & (n - 1) != 0:
        q, _ = torch.linalg.qr(torch.randn(n, n, device=device))
        return q
    H = torch.tensor([[1, 1], [1, -1]], device=device, dtype=torch.float32)
    while H.size(0) < n:
        H = torch.kron(H, torch.tensor([[1, 1], [1, -1]], device=device, dtype=H.dtype))
    H = H[:n, :n] / math.sqrt(n)
    # random sign flipping
    diag = torch.randint(0, 2, (n,), device=device, dtype=torch.float32)
    diag = 2 * diag - 1
    return H * diag


def _quantize_tensor(t: torch.Tensor, bits: int) -> torch.Tensor:
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    scale = t.abs().max().clamp(min=1e-6) / qmax
    q = torch.round(t / scale).clamp(qmin, qmax)
    return q * scale


class CayleySGD(torch.optim.Optimizer):
    """Very small Cayley optimizer for orthonormal matrices."""

    def __init__(self, params: Iterable[nn.Parameter], lr: float = 1.0):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                # project grad onto skew-symmetric
                A = grad @ p.data.t() - p.data @ grad.t()
                update = torch.linalg.inv(
                    torch.eye(p.size(0), device=p.device) + 0.5 * lr * A
                ) @ (torch.eye(p.size(0), device=p.device) - 0.5 * lr * A)
                p.data = update @ p.data
        return loss


class SpinQuant:
    def __init__(self, model: nn.Module, bits: int = 4):
        self.model = model
        self.bits = bits
        self.n_embd = model.config.n_embd
        device = next(model.parameters()).device
        R = _hadamard(self.n_embd, device=device)
        self.R = nn.Parameter(R)

    def _weight_modules(self):
        """Yield linear layers that match the embedding dimension.

        Returns pairs ``(module, mode)`` where ``mode`` indicates whether the
        rotation should be applied on the input dimension (``"in"``) or the
        output dimension (``"out"``).
        """
        for m in self.model.modules():
            if not isinstance(m, nn.Linear):
                continue
            in_dim = m.weight.shape[1]
            out_dim = m.weight.shape[0]
            if in_dim == self.n_embd:
                yield m, "in"
            elif out_dim == self.n_embd:
                yield m, "out"

    def optimize(self, data: Iterable[torch.Tensor], steps: int = 100, lr: float = 1.5):
        """Learn the rotation matrix using simple reconstruction loss."""
        opt = CayleySGD([self.R], lr=lr)
        for _ in range(steps):
            def closure():
                opt.zero_grad()
                loss = 0.0
                R = self.R
                for m, mode in self._weight_modules():
                    W = m.weight
                    if mode == "in":
                        Wr = W @ R.t()
                        Wq = _quantize_tensor(Wr, self.bits)
                        Wrec = Wq @ R
                    else:  # mode == "out"
                        Wr = R @ W
                        Wq = _quantize_tensor(Wr, self.bits)
                        Wrec = R.t() @ Wq
                    loss = loss + F.mse_loss(Wrec, W)
                loss.backward()
                return loss

            opt.step(closure)

    def apply(self):
        with torch.no_grad():
            R = self.R
            for m, mode in self._weight_modules():
                W = m.weight.data
                if mode == "in":
                    Wr = W @ R.t()
                    Wq = _quantize_tensor(Wr, self.bits)
                    m.weight.data = Wq @ R
                else:  # mode == "out"
                    Wr = R @ W
                    Wq = _quantize_tensor(Wr, self.bits)
                    m.weight.data = R.t() @ Wq


