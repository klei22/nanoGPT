import torch


def compute_kurtosis(x: torch.Tensor) -> torch.Tensor:
    """Return kurtosis of a batch of vectors.

    Args:
        x: Tensor of shape (N, D)

    Returns:
        Scalar tensor with mean kurtosis across dimensions.
    """
    mean = x.mean(dim=0, keepdim=True)
    var = x.var(dim=0, unbiased=False, keepdim=True)
    fourth = ((x - mean) ** 4).mean(dim=0)
    kurt = fourth / (var.squeeze(0) ** 2 + 1e-5)
    return kurt.mean()


def learn_kurtail_rotation(acts: torch.Tensor, num_iters: int = 100, lr: float = 1e-2,
                           target: float = 1.8) -> torch.Tensor:
    """Learn an orthogonal rotation that minimizes kurtosis.

    This is a minimal implementation of the rotation learning scheme
    described in the KurTail paper. The function performs gradient
    descent on a rotation matrix so that the transformed activations
    have kurtosis close to the target kurtosis of a uniform distribution
    (approximately 1.8).
    """
    d = acts.shape[-1]
    device = acts.device
    R = torch.eye(d, device=device, dtype=acts.dtype, requires_grad=True)
    opt = torch.optim.Adam([R], lr=lr)
    for _ in range(num_iters):
        transformed = acts @ R
        loss = (compute_kurtosis(transformed) - target) ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
        # project back to the orthogonal group via QR decomposition
        with torch.no_grad():
            q, _ = torch.linalg.qr(R)
            R.copy_(q)
    return R.detach()


def apply_kurtail_quantization(tensor: torch.Tensor, bits: int = 4,
                               num_iters: int = 100) -> tuple:
    """Quantize ``tensor`` after learning a KurTail rotation.

    Returns the zero point, scale and quantized tensor. The learned
    rotation matrix is also returned for reference.
    """
    R = learn_kurtail_rotation(tensor, num_iters=num_iters)
    rotated = tensor @ R
    from .quantize import symmetric_quantize
    zp, scale, q = symmetric_quantize(rotated, bits)
    return zp, scale, q, R

