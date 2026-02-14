import torch
import torch.nn as nn


def unit_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Return ``x`` rescaled to unit L2 norm along ``dim``.

    This helper mirrors the ``Norm`` operation used throughout nGPT, where
    vectors live on the surface of a hypersphere.  An ``eps`` is added for
    numerical stability when a vector has near zero norm.
    """
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def normalize_module_weights(module: nn.Module, embedding_dim: int | None = None) -> None:
    """In-place L2 normalisation of weight tensors within ``module``.

    Any parameter tensor that contains a dimension equal to ``embedding_dim``
    (typically ``config.n_embd``) will be projected onto the unit hypersphere
    along that dimension. Bias terms (``ndim == 1``) are left untouched.

    Parameters
    ----------
    module : nn.Module
        Module whose parameters will be normalised.
    embedding_dim : int, optional
        The embedding dimension ``n_embd``. If ``None`` the function attempts to
        read ``module.config.n_embd`` and falls back to raising ``ValueError`` if
        unavailable.
    """
    if embedding_dim is None:
        embedding_dim = getattr(getattr(module, "config", None), "n_embd", None)
        if embedding_dim is None:
            raise ValueError("embedding_dim must be provided or module.config.n_embd must exist")

    with torch.no_grad():
        for param in module.parameters():
            if param.ndim <= 1:
                continue
            dims = [i for i, s in enumerate(param.shape) if s == embedding_dim]
            for dim in dims:
                param.data = unit_norm(param.data, dim=dim)


__all__ = ["unit_norm", "normalize_module_weights"]
