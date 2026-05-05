import torch
import torch.nn as nn
import torch.nn.functional as F


class EuclideanLMHead(nn.Module):
    """Standard linear language-model head."""

    def __init__(self, in_features: int, vocab_size: int, config):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, in_features))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class HypersphericalLMHead(nn.Module):
    """Cosine-similarity LM head (vectors constrained to a hypersphere)."""

    def __init__(self, in_features: int, vocab_size: int, config):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, in_features))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        self.logit_scale = float(getattr(config, "lm_head_logit_scale", 1.0))
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = F.normalize(x, p=2, dim=-1, eps=self.eps)
        w_n = F.normalize(self.weight, p=2, dim=-1, eps=self.eps)
        return self.logit_scale * F.linear(x_n, w_n)


class HyperbolicLMHead(nn.Module):
    """Poincare-ball classifier: logits are negative squared hyperbolic distances."""

    def __init__(self, in_features: int, vocab_size: int, config):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, in_features))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        self.c = float(getattr(config, "lm_head_hyperbolic_c", 1.0))
        self.logit_scale = float(getattr(config, "lm_head_logit_scale", 1.0))
        self.eps = 1e-6

    def _project_ball(self, x: torch.Tensor) -> torch.Tensor:
        max_norm = (1.0 - self.eps) / (self.c ** 0.5)
        x_norm = x.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        scale = torch.clamp(max_norm / x_norm, max=1.0)
        return x * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ball = self._project_ball(x)
        w_ball = self._project_ball(self.weight)

        x2 = x_ball.pow(2).sum(dim=-1, keepdim=True)  # (B,T,1) or (N,1)
        w2 = w_ball.pow(2).sum(dim=-1).view(*([1] * (x.dim() - 1)), -1)  # (...,V)
        cross = torch.matmul(x_ball, w_ball.transpose(0, 1))  # (...,V)
        sqdist = (x2 + w2 - 2.0 * cross).clamp_min(0.0)

        denom = (1.0 - self.c * x2) * (1.0 - self.c * w2)
        denom = denom.clamp_min(self.eps)
        argument = 1.0 + 2.0 * self.c * sqdist / denom
        argument = argument.clamp_min(1.0 + self.eps)
        d = torch.acosh(argument) / (self.c ** 0.5)
        return -self.logit_scale * d.pow(2)


lm_head_dictionary = {
    "euclidean": EuclideanLMHead,
    "hyperspherical": HypersphericalLMHead,
    "hyperbolic": HyperbolicLMHead,
}
