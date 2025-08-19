# variations/early_exit_router_variants.py
import torch
import torch.nn as nn


class RandomRouter(nn.Module):
    """Router that exits with a fixed probability."""
    def __init__(self, config):
        super().__init__()
        self.exit_prob = getattr(config, 'early_exit_prob', 0.0)

    def forward(self, x: torch.Tensor) -> bool:
        return torch.rand(1, device=x.device).item() < self.exit_prob


class LinearRouter(nn.Module):
    """Learns a score and exits if it exceeds a threshold."""
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.n_embd, 1)
        self.threshold = getattr(config, 'early_exit_threshold', 0.5)

    def forward(self, x: torch.Tensor) -> bool:
        score = self.linear(x.mean(dim=1))
        prob = torch.sigmoid(score)
        return prob.mean().item() > self.threshold


class NormRouter(nn.Module):
    """Exits when the mean L2 norm of activations exceeds a threshold."""
    def __init__(self, config):
        super().__init__()
        self.threshold = getattr(config, 'early_exit_norm_threshold', 1e9)

    def forward(self, x: torch.Tensor) -> bool:
        norm = x.norm(dim=-1).mean()
        return norm.item() > self.threshold


early_exit_router_dictionary = {
    'random': RandomRouter,
    'linear': LinearRouter,
    'norm': NormRouter,
}
