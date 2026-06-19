import numpy as np


def symmetric_quantize(array: np.ndarray, bits: int = 8):
    """Quantize array using symmetric quantization."""
    qmax = (1 << (bits - 1)) - 1
    scale = np.max(np.abs(array)) / float(qmax) if qmax > 0 else 1.0
    if scale == 0:
        scale = 1.0
    zero_point = 0
    q = np.round(array / scale)
    q = np.clip(q, -qmax - 1, qmax).astype(np.int32)
    return q, scale, zero_point


def affine_quantize(array: np.ndarray, bits: int = 8):
    """Quantize array using affine (asymmetric) quantization."""
    qmax = (1 << bits) - 1
    min_val = array.min()
    max_val = array.max()
    scale = (max_val - min_val) / float(qmax) if qmax > 0 else 1.0
    if scale == 0:
        scale = 1.0
    zero_point = np.round(-min_val / scale).astype(np.int32)
    q = np.round(array / scale + zero_point)
    q = np.clip(q, 0, qmax).astype(np.int32)
    return q, scale, zero_point


def dequantize(q: np.ndarray, scale: float, zero_point: int = 0):
    """Dequantize quantized values back to float32."""
    return (q.astype(np.float32) - zero_point) * scale
