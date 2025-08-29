import torch
import torch.nn.functional as F


def add_residual(
    x: torch.Tensor,
    update: torch.Tensor,
    alpha: float = 1.0,
    *args,
    **kwargs,
) -> torch.Tensor:
    """Standard residual connection using addition."""
    return x + update


def lerp_residual(
    x: torch.Tensor,
    update: torch.Tensor,
    alpha: float = 1.0,
    *args,
    **kwargs,
) -> torch.Tensor:
    """Linear interpolation between ``x`` and ``x + update`` without ``torch.lerp``."""
    return x + alpha * update


def slerp_residual(
    x: torch.Tensor,
    update: torch.Tensor,
    alpha: float = 1.0,
    threshold: float = 1e-7,
    use_lerp_fallback: bool = False,
) -> torch.Tensor:
    """Spherical linear interpolation between ``x`` and ``x + update``.

    Parameters
    ----------
    x, update : torch.Tensor
        Tensors to interpolate between.
    alpha : float or Tensor
        Interpolation factor.
    threshold : float or Tensor
        If ``|sin(omega)|`` falls below this value, ``use_lerp_fallback`` controls behaviour.
    use_lerp_fallback : bool
        If True, falls back to linear interpolation when the angle is small. Otherwise the
        denominator is clamped to ``threshold``.
    """
    end = x + update
    start_norm = F.normalize(x, dim=-1)
    end_norm = F.normalize(end, dim=-1)
    dot = (start_norm * end_norm).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    so = torch.sin(omega)

    alpha_t = torch.as_tensor(alpha, dtype=x.dtype, device=x.device)
    while alpha_t.dim() < omega.dim():
        alpha_t = alpha_t.view(*(1 for _ in range(alpha_t.dim())), 1)
    alpha_t = alpha_t.expand_as(omega)

    thr_t = torch.as_tensor(threshold, dtype=x.dtype, device=x.device)
    while thr_t.dim() < so.dim():
        thr_t = thr_t.view(*(1 for _ in range(thr_t.dim())), 1)
    small = so.abs() <= thr_t
    safe_so = torch.where(small, thr_t, so)

    interp = (torch.sin((1 - alpha_t) * omega) / safe_so) * x + (
        torch.sin(alpha_t * omega) / safe_so
    ) * end

    if use_lerp_fallback:
        lerp_out = lerp_residual(x, update, alpha)
        return torch.where(small.expand_as(interp), lerp_out, interp)
    else:
        return interp


residual_dictionary = {
    "add": add_residual,
    "lerp": lerp_residual,
    "slerp": slerp_residual,
}

