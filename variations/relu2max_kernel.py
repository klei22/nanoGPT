"""Inductor-compiled implementation of the elementwise ReLU2Max transform."""

import torch


def _relu2max(x: torch.Tensor, divisor: float, sequence_length: int) -> torch.Tensor:
    """Keep this function small so Inductor can emit one pointwise kernel."""
    return torch.relu(x).square() / (divisor * sequence_length)


# Compilation is lazy: importing nanoGPT does not initialize a compiler or CUDA.
# AOTAutograd also derives a compiled backward kernel from this function.
relu2max_kernel = torch.compile(_relu2max, fullgraph=True, dynamic=True)
