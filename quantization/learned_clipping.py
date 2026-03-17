# quantization/learned_clipping.py
"""
QAT with learned clip values (PACT/LSQ-style) for outlier channel mitigation.

Reference: Nrusimha et al., "Mitigating the Impact of Outlier Channels for
Language Model Quantization with Activation Regularization" (arXiv:2404.03605)

Implements learned per-layer clipping values c_minus and c_plus that are
optimized via gradient descent using the straight-through estimator (STE).
"""

import torch
import torch.nn as nn


class LearnedClippingQAT(nn.Module):
    """Per-layer QAT module with learnable clip values.

    During the forward pass the input is clamped to [c_minus, c_plus],
    uniformly quantized to ``bits`` levels, and immediately dequantized
    (fake quantization).  Gradients flow through via the STE with
    analytically-correct gradients for the clip parameters.
    """

    def __init__(self, bits: int = 4, init_clip: float = 4.0, align_zero: bool = True):
        super().__init__()
        self.bits = bits
        self.align_zero = align_zero
        # Learnable clip boundaries (Algorithm 1 in the paper)
        self.c_plus = nn.Parameter(torch.tensor(init_clip))
        self.c_minus = nn.Parameter(torch.tensor(-init_clip))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            # At inference, still quantize/dequantize for consistent behavior
            pass
        return _LearnedClipFunction.apply(
            x, self.c_minus, self.c_plus, self.bits, self.align_zero
        )


class _LearnedClipFunction(torch.autograd.Function):
    """Algorithms 1 & 2 from the paper: forward quantize/dequantize and
    STE backward with clip-value gradients."""

    @staticmethod
    def forward(ctx, x, c_minus, c_plus, bits, align_zero):
        n_levels = (1 << bits) - 1  # 2^b - 1
        s = n_levels / (c_plus - c_minus)

        if align_zero:
            z = torch.round(s * c_minus)
        else:
            z = torch.zeros(1, device=x.device)

        # Quantize (Algorithm 1)
        clamped = x.clamp(c_minus.item(), c_plus.item())
        q = torch.round(s * clamped + z)
        # Dequantize
        x_hat = (q - z) / s

        # Save for backward
        ctx.save_for_backward(x, c_minus, c_plus)
        ctx.s = s
        ctx.n_levels = n_levels
        return x_hat

    @staticmethod
    def backward(ctx, grad_output):
        x, c_minus, c_plus = ctx.saved_tensors
        s = ctx.s
        n_levels = ctx.n_levels

        # Masks for STE (Algorithm 2)
        below = x < c_minus
        above = x > c_plus
        in_range = ~below & ~above

        # Gradient w.r.t. input: STE passes through for in-range values
        grad_x = grad_output * in_range.float()

        # Quantization error term for clip gradients
        q_scaled = s * (x - c_minus)
        error = (q_scaled - torch.round(q_scaled)) / n_levels

        # Gradient w.r.t. c_plus (Algorithm 2)
        grad_c_plus = torch.zeros_like(grad_output)
        grad_c_plus[above] = grad_output[above]
        grad_c_plus[in_range] = -error[in_range] * grad_output[in_range]
        grad_c_plus = grad_c_plus.sum()

        # Gradient w.r.t. c_minus (Algorithm 2)
        grad_c_minus = torch.zeros_like(grad_output)
        grad_c_minus[below] = grad_output[below]
        # For in-range values where x < c_plus (essentially all in-range)
        grad_c_minus[in_range] = -error[in_range] * grad_output[in_range]
        grad_c_minus = grad_c_minus.sum()

        return grad_x, grad_c_minus, grad_c_plus, None, None


def compute_kurtosis(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute the kurtosis of activations summed over tokens.

    Following the paper: Kurtosis(d) = sum_i ((x_i - mu)^4) / (sigma^4 + eps)
    Computed per-token then summed, which penalizes heavy-tailed output
    distributions and discourages pathologically large weight rows.
    """
    # x shape: (B, T, D) — compute stats over the feature dimension
    mu = x.mean(dim=-1, keepdim=True)
    diff = x - mu
    var = (diff ** 2).mean(dim=-1, keepdim=True)
    kurt = ((diff ** 4).mean(dim=-1, keepdim=True)) / (var ** 2 + eps)
    return kurt.sum()
