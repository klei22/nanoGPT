import math
import copy
import torch


def _sign_matrix(out_dim: int, in_dim: int, generator: torch.Generator) -> torch.Tensor:
    mat = torch.randint(0, 2, (out_dim, in_dim), generator=generator, dtype=torch.float32)
    mat = mat * 2 - 1
    mat /= math.sqrt(out_dim)
    return mat


def _gaussian_matrix(out_dim: int, in_dim: int, generator: torch.Generator,
                      mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    mat = torch.empty(out_dim, in_dim, dtype=torch.float32)
    mat.normal_(mean=mean, std=std, generator=generator)
    mat /= math.sqrt(out_dim)
    return mat


def _sparse_matrix(out_dim: int, in_dim: int, generator: torch.Generator) -> torch.Tensor:
    rand = torch.rand(out_dim, in_dim, generator=generator)
    mat = torch.zeros_like(rand)
    mat[rand < 1/6] = 1
    mat[(rand >= 1/6) & (rand < 2/6)] = -1
    mat *= math.sqrt(3.0 / out_dim)
    return mat


def _srht_matrix(out_dim: int, in_dim: int, generator: torch.Generator) -> torch.Tensor:
    if (in_dim & (in_dim - 1)) != 0:
        raise ValueError("srht JL requires input dim to be a power of two")
    def hadamard(n: int) -> torch.Tensor:
        H = torch.tensor([[1.0]])
        size = 1
        while size < n:
            H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
            size *= 2
        return H
    D = torch.randint(0, 2, (in_dim,), generator=generator, dtype=torch.float32) * 2 - 1
    H = hadamard(in_dim)
    idx = torch.randperm(in_dim, generator=generator)[:out_dim]
    proj = H[idx] * D
    proj /= math.sqrt(out_dim)
    return proj


def _qr_matrix(out_dim: int, in_dim: int, generator: torch.Generator) -> torch.Tensor:
    raw = torch.randn(out_dim, in_dim, generator=generator)
    q, _ = torch.linalg.qr(raw.T, mode="reduced")
    proj = q.T
    proj /= math.sqrt(out_dim)
    return proj


def build_jl_matrix(out_dim: int, in_dim: int, jl_type: str, generator: torch.Generator,
                    mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    if jl_type == "gaussian":
        return _gaussian_matrix(out_dim, in_dim, generator, mean, std)
    if jl_type == "sign":
        return _sign_matrix(out_dim, in_dim, generator)
    if jl_type == "sparse":
        return _sparse_matrix(out_dim, in_dim, generator)
    if jl_type == "srht":
        return _srht_matrix(out_dim, in_dim, generator)
    if jl_type == "qr":
        return _qr_matrix(out_dim, in_dim, generator)
    raise ValueError(f"Unsupported jl_type: {jl_type}")


def jl_project_tensor(tensor: torch.Tensor, proj: torch.Tensor,
                      vertical_only: bool = False) -> torch.Tensor:
    in_dim = proj.shape[1]
    if tensor.ndim == 0:
        return tensor
    if vertical_only:
        if tensor.ndim > 1 and tensor.shape[0] == in_dim:
            return proj @ tensor
        if tensor.ndim == 1 and tensor.shape[0] == in_dim:
            return (proj @ tensor.unsqueeze(-1)).squeeze(-1)
        return tensor
    if tensor.ndim >= 1 and tensor.shape[-1] == in_dim:
        tensor = tensor @ proj.t()
    if tensor.ndim > 1 and tensor.shape[0] == in_dim:
        tensor = proj @ tensor
    elif tensor.ndim == 1 and tensor.shape[0] == in_dim:
        tensor = (proj @ tensor.unsqueeze(-1)).squeeze(-1)
    return tensor


def jl_transform_model(model: torch.nn.Module, out_embd: int | None = None,
                       jl_type: str = "gaussian", seed: int = 1337,
                       cproj_vertical: bool = False, mean: float = 0.0,
                       std: float = 1.0) -> torch.nn.Module:
    old_embd = getattr(getattr(model, "config", None), "n_embd", None)
    if old_embd is None:
        raise ValueError("model.config.n_embd is required")
    out_embd = old_embd if out_embd is None else out_embd
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    proj = build_jl_matrix(out_embd, old_embd, jl_type, g, mean, std)
    state_dict = model.state_dict()
    new_state = {}
    for key, tensor in state_dict.items():
        if not torch.is_floating_point(tensor):
            new_state[key] = tensor
            continue
        vertical = cproj_vertical and key.endswith("c_proj.weight")
        new_state[key] = jl_project_tensor(tensor.cpu(), proj, vertical_only=vertical)
    if out_embd != old_embd:
        cfg = copy.deepcopy(model.config)
        cfg.n_embd = out_embd
        if hasattr(cfg, "n_qk_head_dim"):
            cfg.n_qk_head_dim = getattr(model.config, "n_qk_head_dim", None)
        if hasattr(cfg, "n_v_head_dim"):
            cfg.n_v_head_dim = getattr(model.config, "n_v_head_dim", None)
        new_model = type(model)(cfg)
        new_model.load_state_dict(new_state, strict=False)
        return new_model
    model.load_state_dict(new_state)
    return model


__all__ = [
    "build_jl_matrix",
    "jl_project_tensor",
    "jl_transform_model",
]
