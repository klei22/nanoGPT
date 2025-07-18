# utils/model_stats.py
import torch
from torch import Tensor
from typing import Dict, Tuple

def _moments(t: Tensor) -> Dict[str, float]:
    """
    In‑GPU computation of stdev, kurtosis, min, max, abs_max.
    Returns python floats (so we detach only tiny scalars).
    """
    # keep in fp32 for numeric stability
    t_f32 = t.float()
    mean     = torch.mean(t_f32)
    var      = torch.var(t_f32, unbiased=False)
    stdev    = torch.sqrt(var)

    # Fisher kurtosis (zero for N(0,1))
    m4       = torch.mean((t_f32 - mean) ** 4)
    kurtosis = m4 / var**2 - 3.0

    t_min    = torch.min(t_f32)
    t_max    = torch.max(t_f32)
    abs_max  = torch.max(torch.abs(t_f32))

    return dict(
        stdev    = stdev.item(),
        kurtosis = kurtosis.item(),
        max      = t_max.item(),
        min      = t_min.item(),
        abs_max  = abs_max.item(),
    )

@torch.no_grad()
def compute_weight_stats(model: torch.nn.Module, device: torch.device) -> Tuple[Dict[str, Dict], Dict[str, float]]:
    per_tensor: Dict[str, Dict] = {}
    accum      = {k:0.0 for k in ["stdev","kurtosis","max","min","abs_max"]}
    n          = 0

    for name, p in model.named_parameters():
        if p.requires_grad:                       # skip buffers etc.
            t = p.detach().to(device)             # stays on GPU if asked
            s = _moments(t)
            per_tensor[name] = s
            for k in accum:                       # running mean over tensors
                accum[k] += s[k]
            n += 1

    overall = {k: v/n for k,v in accum.items()}
    return per_tensor, overall

@torch.no_grad()
def compute_activation_stats(
    model:      torch.nn.Module,
    x:          Tensor,
    y:          Tensor,
    iter_num:   int,
    device:     torch.device = torch.device("cpu"),
) -> Tuple[Dict[str, Dict], Dict[str, float]]:
    """
    One‑off activation scan used inside `Trainer.estimate_loss`.
    Performs a *single* forward pass with temporary hooks that:  
      • run on the requested `device` (GPU keeps host RAM flat)  
      • compute moments on‑the‑fly and immediately discard the tensor  
    Returns a dict keyed by module‑path and an overall average.
    """
    act_stats: Dict[str, Dict] = {}
    overall   = {k: 0.0 for k in ["stdev", "kurtosis", "max", "min", "abs_max"]}
    n         = 0

    def make_hook(mod_name: str):
        def _hook(_module, _inp, out):
            nonlocal n
            # Work with first tensor output if module returns tuple
            t = out[0] if isinstance(out, (tuple, list)) else out
            if not torch.is_tensor(t):
                return                          # skip non‑tensor outputs
            s = _moments(t.detach().to(device))
            act_stats[mod_name] = s
            for k in overall:
                overall[k] += s[k]
            n += 1
            # free ASAP
            del t
        return _hook

    # Register hooks only on *leaf* modules to avoid duplication
    handles = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:   # leaf
            handles.append(module.register_forward_hook(make_hook(name)))

    # Forward pass (targets are optional; model may ignore them)
    _ = model(x, y, iter_num=iter_num) if y is not None else model(x)

    # Clean up
    for h in handles:
        h.remove()

    # Guard against empty hook collection
    if n == 0:
        return {}, {k: 0.0 for k in overall}

    overall = {k: v / n for k, v in overall.items()}
    return act_stats, overall
