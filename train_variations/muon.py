"""Muon optimiser implementation.

Implementation of the Muon optimiser from Jordan et al. (2024), which
orthogonalises SGD-momentum updates using a quintic Newton–Schulz
iteration.  This file derives from the MIT-licensed reference code:

@misc{jordan2024muon,
  author       = {Keller Jordan and Yuchen Jin and Vlado Boza and You Jiacheng and
                  Franz Cesista and Laker Newhouse and Jeremy Bernstein},
  title        = {Muon: An optimizer for hidden layers in neural networks},
  year         = {2024},
  url          = {https://kellerjordan.github.io/posts/muon/}
}
"""

import torch
import torch.distributed as dist


POLAR_EXPRESS_COEFFS = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]


def _polarexpress_coeffs(steps: int):
    """Return the coefficient triples used by the Polar Express iteration."""

    if steps <= 0:
        return []

    coeffs = POLAR_EXPRESS_COEFFS
    scaled = [
        (a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in coeffs[:-1]
    ] + [coeffs[-1]]
    if steps <= len(scaled):
        return scaled[:steps]
    return scaled + [scaled[-1]] * (steps - len(scaled))


def zeropower_via_newtonschulz5(G, steps: int):
    """Orthogonalize matrix ``G`` via a quintic Newton--Schulz iteration."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def zeropower_via_polarexpress(G, steps: int):
    """Orthogonalize matrix ``G`` using the Polar Express iteration."""

    assert G.ndim >= 2
    coeffs = _polarexpress_coeffs(steps)
    if not coeffs:
        return G

    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    denom = X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7
    X = X / denom

    coeff_tensors = [
        (X.new_tensor(a), X.new_tensor(b), X.new_tensor(c)) for a, b, c in coeffs
    ]
    for a, b, c in coeff_tensors:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.mT
    return X


def muon_update(
    grad,
    momentum,
    beta=0.95,
    ns_steps=5,
    nesterov=True,
    orthogonalizer: str = "jordan",
):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    if orthogonalizer == "polar_express":
        update = zeropower_via_polarexpress(update, steps=ns_steps)
    elif orthogonalizer == "jordan":
        update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    else:
        raise ValueError(f"Unknown Muon orthogonalizer: {orthogonalizer}")
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


class Muon(torch.optim.Optimizer):
    """Distributed Muon optimiser for 2D hidden-layer weights."""
    def __init__(
        self,
        params,
        lr=0.02,
        weight_decay=0,
        momentum=0.95,
        orthogonalizer: str = "jordan",
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            orthogonalizer=orthogonalizer,
        )
        assert isinstance(params, list) and params and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = group["params"]
            pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
            for base in range(0, len(params), dist.get_world_size()):
                if base + dist.get_rank() < len(params):
                    p = params[base + dist.get_rank()]
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    upd = muon_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        orthogonalizer=group.get("orthogonalizer", "jordan"),
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(upd.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(pad[base:base + dist.get_world_size()], pad[base + dist.get_rank()])
        return loss


class SingleDeviceMuon(torch.optim.Optimizer):
    """Muon variant for single-device training."""
    def __init__(
        self,
        params,
        lr=0.02,
        weight_decay=0,
        momentum=0.95,
        orthogonalizer: str = "jordan",
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            orthogonalizer=orthogonalizer,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                state = self.state[p]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                upd = muon_update(
                    p.grad,
                    state["momentum_buffer"],
                    beta=group["momentum"],
                    orthogonalizer=group.get("orthogonalizer", "jordan"),
                )
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(upd.reshape(p.shape), alpha=-group["lr"])
        return loss


# --- Helpers for mixed parameter sets ------------------------------------

def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    b1c = buf1 / (1 - betas[0]**step)
    b2c = buf2 / (1 - betas[1]**step)
    return b1c / (b2c.sqrt() + eps)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Muon with auxiliary Adam for non-matrix parameters."""
    def __init__(self, param_groups):
        for g in param_groups:
            assert "use_muon" in g
            if g["use_muon"]:
                g.setdefault("lr", 0.02)
                g.setdefault("momentum", 0.95)
                g.setdefault("weight_decay", 0)
                g.setdefault("orthogonalizer", "jordan")
            else:
                g.setdefault("lr", 3e-4)
                g.setdefault("betas", (0.9, 0.95))
                g.setdefault("eps", 1e-10)
                g.setdefault("weight_decay", 0)
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base in range(0, len(params), dist.get_world_size()):
                    if base + dist.get_rank() < len(params):
                        p = params[base + dist.get_rank()]
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if not state:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        upd = muon_update(
                            p.grad,
                            state["momentum_buffer"],
                            beta=group["momentum"],
                            orthogonalizer=group.get("orthogonalizer", "jordan"),
                        )
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(upd.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(pad[base:base + dist.get_world_size()], pad[base + dist.get_rank()])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if not state:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    upd = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"], state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(upd, alpha=-group["lr"])
        return loss


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """Single-device variant of :class:`MuonWithAuxAdam`."""
    def __init__(self, param_groups):
        for g in param_groups:
            assert "use_muon" in g
            if g["use_muon"]:
                g.setdefault("lr", 0.02)
                g.setdefault("momentum", 0.95)
                g.setdefault("weight_decay", 0)
                g.setdefault("orthogonalizer", "jordan")
            else:
                g.setdefault("lr", 3e-4)
                g.setdefault("betas", (0.9, 0.95))
                g.setdefault("eps", 1e-10)
                g.setdefault("weight_decay", 0)
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    upd = muon_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        orthogonalizer=group.get("orthogonalizer", "jordan"),
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(upd.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if not state:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    upd = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"], state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(upd, alpha=-group["lr"])
        return loss
