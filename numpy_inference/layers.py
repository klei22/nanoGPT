import numpy as np
from .quantization import dequantize, symmetric_quantize


class Linear:
    def __init__(self, weight: np.ndarray, bias: np.ndarray | None = None,
                 quant_bits: int | None = None):
        self.bias = bias
        if quant_bits is not None:
            self.qweight, self.scale, self.zero_point = symmetric_quantize(weight, quant_bits)
            self.weight = None
            self.quantized = True
        else:
            self.weight = weight.astype(np.float32)
            self.quantized = False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.quantized:
            w = dequantize(self.qweight, self.scale, self.zero_point)
        else:
            w = self.weight
        y = x @ w.T
        if self.bias is not None:
            y += self.bias
        return y


class LayerNorm:
    def __init__(self, weight: np.ndarray, bias: np.ndarray | None = None, eps: float = 1e-5):
        self.weight = weight
        self.bias = bias
        self.eps = eps

    def __call__(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        y = (x - mean) / np.sqrt(var + self.eps)
        y = y * self.weight
        if self.bias is not None:
            y += self.bias
        return y


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x ** 3))))
