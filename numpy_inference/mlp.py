import numpy as np
from .layers import Linear, gelu


class MLP:
    def __init__(self, up_weight, down_weight, up_bias=None, down_bias=None,
                 quant_bits=None):
        self.fc = Linear(up_weight, up_bias, quant_bits)
        self.proj = Linear(down_weight, down_bias, quant_bits)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = self.fc(x)
        x = gelu(x)
        x = self.proj(x)
        return x
