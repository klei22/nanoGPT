import torch
from torch import nn
from typing import Iterable, Tuple

class DummyOptimizer:
    def step(self):
        pass
    def zero_grad(self, set_to_none: bool = True):
        pass
    def state_dict(self):
        return {}
    def load_state_dict(self, state):
        pass

def mezo_step(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor,
              lr: float, epsilon: float) -> Tuple[float, float]:
    """Perform one MeZO update step.

    Returns:
        loss_pos (float): loss with positive perturbation.
        loss_neg (float): loss with negative perturbation.
    """
    noises: Iterable[torch.Tensor] = []
    for p in model.parameters():
        z = torch.randn_like(p)
        p.data.add_(epsilon * z)
        noises.append(z)
    logits_pos, loss_pos = model(inputs, targets=targets)
    for p, z in zip(model.parameters(), noises):
        p.data.add_(-2 * epsilon * z)
    logits_neg, loss_neg = model(inputs, targets=targets)
    for p, z in zip(model.parameters(), noises):
        p.data.add_(epsilon * z)
    g_scalar = (loss_pos - loss_neg) / (2 * epsilon)
    for p, z in zip(model.parameters(), noises):
        p.data.add_( -lr * g_scalar * z )
    return float(loss_pos), float(loss_neg)
