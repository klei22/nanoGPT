"""Low-memory, differentiable diagnostics for predictive-width experiments."""
import torch


def participation_dimension(features, eps=1e-12):
    x = features.reshape(-1, features.shape[-1]).float()
    x = x - x.mean(0, keepdim=True)
    singular_values = torch.linalg.svdvals(x)
    eigenvalues = singular_values.square() / max(1, x.shape[0] - 1)
    return eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(eps)


def jensen_gaps(logits, eps=1e-12):
    """Return exact and Hessian-quadratic gaps; ensemble axis is dimension 0."""
    mean = logits.mean(0)
    delta = logits - mean
    probabilities = mean.softmax(-1)
    exact = torch.logsumexp(logits, -1).mean(0) - torch.logsumexp(mean, -1)
    mean_delta = (delta * probabilities).sum(-1)
    quadratic = 0.5 * ((delta.square() * probabilities).sum(-1) - mean_delta.square()).mean(0)
    return {"exact": exact, "quadratic": quadratic, "residual": exact - quadratic}


def hessian_inner(a, b, probabilities):
    return (probabilities * a * b).sum(-1) - (probabilities * a).sum(-1) * (probabilities * b).sum(-1)


def hessian_branch_correlation(branch_logits, probabilities=None, eps=1e-12):
    """Correlation matrix and effective stream count for ``[...,S,V]`` logits."""
    if probabilities is None:
        probabilities = branch_logits.sum(-2).softmax(-1)
    centered = branch_logits - branch_logits.mean(tuple(range(branch_logits.ndim - 2)), keepdim=True)
    gram = torch.einsum("...sv,...tv,...v->...st", centered, centered, probabilities)
    gram = gram.reshape(-1, *gram.shape[-2:]).mean(0)
    scale = gram.diag().clamp_min(eps).sqrt()
    corr = gram / (scale[:, None] * scale[None, :])
    s = corr.shape[0]
    rho = (corr.sum() - corr.diag().sum()) / max(1, s * (s - 1))
    return {"correlation": corr, "spectrum": torch.linalg.eigvalsh(corr),
            "mean_off_diagonal": rho, "effective_streams": s / (1 + (s - 1) * rho).clamp_min(eps)}


def effective_weighted_samples(weights, eps=1e-12):
    return weights.sum(dim=tuple(range(weights.ndim - 1))).square() / weights.square().sum(
        dim=tuple(range(weights.ndim - 1))).clamp_min(eps)
