# variations/linear_variations.py
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Optional
from .activation_variations import *
from functools import lru_cache
from quantization.quantize import _fake_quantize, quantize_dictionary, dequantize

class WrappedLinear(nn.Linear):
    """ Adapts nn.Linear to add 'config' parameter for interface polymorphism"""
    def __init__(self, in_features, out_features, config=None, method=None, bits=None, bias=None):
        super(WrappedLinear, self).__init__(in_features, out_features, bias)

class QuantizedLinear(nn.Linear):
    """Linear layer with quantization aware training capability
    Source: https://github.com/Alexstrasza98/Transformer-Quantization/blob/main
    Source License: MIT
    """

    def __init__(self, in_features, out_features, config=None, method="affine_quant", bits=8, bias=True):
        super().__init__(in_features, out_features, bias)

        self.weight_bits = bits
        self.quant_method = method
        self.start_quant_level = config.start_quant_level
        self.quant_scheduler = config.quant_scheduler
        self.full_quant_iteration = config.full_quant_iteration
        self.eval_interval = config.eval_interval

        if self.weight_bits < 1:
            raise ValueError(f"weight_bits={self.weight_bits} must be higher than 0 ")

        self.warmup_step = config.quantization_warmup_iters
        self.accumulation_bits = 32

        # Placeholder for quantized weights during training
        self._fake_quantized_weight = None
        if bias == True:
            self.register_buffer("quantized_bias", None)
            self.register_buffer("bias_norm", None)
            self.register_buffer("bias_zero_point", torch.tensor([0]))

        self.register_buffer("_step", torch.zeros(1))

        self.register_buffer("quantized_weight", None)
        self.register_buffer("weight_norm", None)
        self.register_buffer("weight_zero_point", torch.tensor([0]))

    def training_quantized_forward(self, input):
        """Fake quantizes weights. Function should only be used while training"""
        assert self.training, "Should be called only during training"

        # Applies the fake quantization to the weights
        self._fake_quantized_weight = _fake_quantize(self.weight, self.training, self.quant_scheduler, self.start_quant_level, self.full_quant_iteration, self.eval_interval, self._step.item(), self.weight_bits, self.quant_method)
        # Uses the quantized weights to compute the output using F.linear
        out = F.linear(input, self._fake_quantized_weight, self.bias)

        return out

    def inference_quantized_forward(self, input):
        """Simulate quantized inference. Function should be called only during inference"""
        assert not self.training, "Should be called only during inference"

        # Compute the dequantized weight
        weight = dequantize(self.weight_zero_point[0], self.weight_norm, self.quantized_weight)

        # Compute the dequantized bias
        if self.bias is not None:
            bias = dequantize(self.bias_zero_point[0], self.bias_norm, self.quantized_bias)

        # Uses the dequantized weights and bias to compute the output using F.linear
        if self.bias:
            out = F.linear(input, weight, bias)
        else:
            out = F.linear(input, weight)

        return out

    def _eval(self):
        """Sets the model for inference by quantizing the model"""
        self.weight_zero_point[0], self.weight_norm, self.quantized_weight = quantize_dictionary[self.quant_method](self.weight, self.weight_bits)

        if self.bias is not None:
            self.bias_zero_point[0], self.bias_norm, self.quantized_bias = quantize_dictionary[self.quant_method](self.bias, self.accumulation_bits)

    def forward(self, input):
        """Passes the input through the model during training and inference"""
        if self.training:
            if self._step > self.warmup_step:
                out = self.training_quantized_forward(input)
            else:
                out = super().forward(input)
            self._step += 1
        else:
            # Prepares the model for inference by quantizing weights and bias
            self._eval()
            # Uses quantized weights and bias to compute the output
            out = self.inference_quantized_forward(input)
        return out


class AdaptiveBitLinear(nn.Linear):
    """Linear layer with learnable bit-width using STE fake quantization."""

    def __init__(
        self,
        in_features,
        out_features,
        config=None,
        method=None,
        bits=None,
        bias=True,
        min_bits: Optional[float] = None,
        max_bits: Optional[float] = None,
        activation_bits: Optional[float] = None,
        quantize_input: Optional[bool] = None,
    ):
        super().__init__(in_features, out_features, bias)

        default_min_bits = 1.0
        default_max_bits = 8.0
        default_activation_bits = 8.0
        default_quantize_input = True

        if config is not None:
            default_min_bits = getattr(config, "adaptive_linear_min_bits", default_min_bits)
            default_max_bits = getattr(config, "adaptive_linear_max_bits", default_max_bits)
            default_activation_bits = getattr(
                config, "adaptive_linear_activation_bits", default_activation_bits
            )
            default_quantize_input = getattr(
                config, "adaptive_linear_quantize_input", default_quantize_input
            )
            if bits is None:
                bits = getattr(config, "adaptive_linear_init_bits", None)

        self.min_bits = float(default_min_bits if min_bits is None else min_bits)
        self.max_bits = float(default_max_bits if max_bits is None else max_bits)
        self.quantize_input = default_quantize_input if quantize_input is None else quantize_input
        self.activation_bits = float(
            default_activation_bits if activation_bits is None else activation_bits
        )

        init_bits = float(bits if bits is not None else self.max_bits)
        init_bits = max(self.min_bits, min(init_bits, self.max_bits))
        self.bit_param = nn.Parameter(torch.tensor(init_bits, dtype=torch.float32))

        self.quant_method = method
        self.register_buffer("_last_bitwidth", torch.tensor(init_bits, dtype=torch.float32))

    @staticmethod
    def _ste_round(value: torch.Tensor) -> torch.Tensor:
        return value + (torch.round(value) - value).detach()

    def current_bitwidth(self) -> torch.Tensor:
        clipped = torch.clamp(self.bit_param, self.min_bits, self.max_bits)
        bitwidth = self._ste_round(clipped)
        self._last_bitwidth = bitwidth.detach()
        return bitwidth

    def _fake_quantize_tensor(self, tensor: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        orig_dtype = tensor.dtype
        tensor_fp32 = tensor.to(torch.float32)
        bits_fp32 = bits.to(tensor_fp32.dtype)

        levels = torch.pow(torch.tensor(2.0, device=tensor_fp32.device), bits_fp32 - 1.0)
        qmax = torch.clamp(levels - 1.0, min=1.0)
        qmin = -levels

        max_val = tensor_fp32.abs().max()
        scale = max_val / qmax
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)

        quantized = torch.clamp(torch.round(tensor_fp32 / scale), qmin, qmax) * scale
        return (tensor_fp32 + (quantized - tensor_fp32).detach()).to(orig_dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bitwidth = self.current_bitwidth()
        weight = self._fake_quantize_tensor(self.weight, bitwidth)

        if self.quantize_input:
            input_bits = torch.tensor(self.activation_bits, device=input.device, dtype=bitwidth.dtype)
            input = self._fake_quantize_tensor(input, input_bits)

        return F.linear(input, weight, self.bias)

    def bit_usage(self) -> torch.Tensor:
        bits = self.current_bitwidth()
        total = bits * self.weight.numel()
        if self.bias is not None:
            total = total + bits * self.bias.numel()
        return total

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"min_bits={self.min_bits}, max_bits={self.max_bits}, "
            f"quantize_input={self.quantize_input}, activation_bits={self.activation_bits}"
        )


class BitLinear1p58(nn.Linear):
    """ BitLinear from Era of 1.58 LLMs Paper
    Source: https://huggingface.co/1bitLLM/bitnet_b1_58-large/blob/main/utils_quant.py
    Source License: MIT
    Paper Link: https://arxiv.org/abs/2402.17764
    """

    def __init__(self, in_features, out_features, config=None, method=None, bits=None, bias=True, num_groups=1):
        super().__init__(in_features, out_features, bias)

        """
        RMSNorm is placed outside BitLinear
        """
        weight_bits=1
        input_bits=8
        self.weight_bits = weight_bits
        self.input_bits = input_bits

    def forward(self, x):

        quant_input = x + (self.activation_quant(x, self.input_bits) - x).detach()
        quant_weight = self.weight + (self.weight_quant(self.weight, self.weight_bits) - self.weight).detach()

        out = nn.functional.linear(quant_input, quant_weight)
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out

    def weight_quant(self, weight, num_bits=1):
        dtype = weight.dtype
        weight = weight.float()
        s =  1 / weight.abs().mean().clamp(min=1e-5)
        result = (weight * s).round().clamp(-1, 1) / s
        return result.type(dtype)

    def activation_quant(self, x, num_bits=8):
        dtype = x.dtype
        x = x.float()
        Qn = -2 ** (num_bits - 1)
        Qp = 2 ** (num_bits - 1) - 1
        s = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        result = (x * s).round().clamp(Qn, Qp) / s
        return result.type(dtype)

class BitLinear(nn.Linear):
    """PyTorch BitLinear Layer
    Source: https://github.com/Beomi/BitNet-Transformers/tree/main
    Source License: Apache Version 2.0
    """

    def __init__(self, in_features, out_features, config=None, method=None, bits=None, bias=True, num_groups=1):
        super(BitLinear, self).__init__(in_features, out_features, bias)
        self.num_groups = num_groups
        self.eps = 1e-5

    def ste_binarize(self, x):
        # Apply the sign function for binarization
        binarized_x = torch.sign(x)
        # Use STE: during backward pass, we bypass the binarization
        binarized_x = (binarized_x - x).detach() + x
        return binarized_x

    def binarize_weights_groupwise(self):
        # Divide weights into groups
        group_size = self.weight.shape[0] // self.num_groups
        binarized_weights = torch.zeros_like(self.weight)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            weight_group = self.weight[start_idx:end_idx]

            # Binarize each group using STE
            alpha_g = weight_group.mean()
            binarized_weights[start_idx:end_idx] = self.ste_binarize(
                weight_group - alpha_g
            )

        return binarized_weights

    def quantize_activations_groupwise(self, x, b=8):
        Q_b = 2 ** (b - 1)

        # Divide activations into groups
        group_size = x.shape[0] // self.num_groups
        quantized_x = torch.zeros_like(x)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            activation_group = x[start_idx:end_idx]

            # Quantize each group
            gamma_g = activation_group.abs().max()
            quantized_x[start_idx:end_idx] = torch.clamp(
                activation_group * Q_b / (gamma_g + self.eps),
                -Q_b + self.eps,
                Q_b - self.eps,
            )

        return quantized_x

    def forward(self, input):
        # Binarize weights (group-wise) using STE
        binarized_weights = self.binarize_weights_groupwise()

        # Normal linear transformation with binarized weights
        output = torch.nn.functional.linear(input, binarized_weights, self.bias)

        # Quantize activations group-wise
        output = self.quantize_activations_groupwise(output)

        return output


class BitLinearOptimized(nn.Linear):
    """Memory Optimized BitLinear Layer
    Source: https://github.com/Beomi/BitNet-Transformers/tree/main
    Source License: Apache Version 2.0
    """

    def __init__(self, in_features, out_features, config=None, method=None, bits=None, bias=True, num_groups=1):
        super(BitLinearOptimized, self).__init__(in_features, out_features, bias)
        self.num_groups = num_groups
        self.eps = 1e-5

        # Initialize 1-bit quantized weights and store them as int8
        self.register_buffer(
            "quantized_weights", torch.sign(self.weight.data).to(torch.int8)
        )
        # Clear the original weights to save memory
        del self.weight

    @property
    def weight(self):
        # Return the dequantized weights when accessed
        return self.dequantize_weights()

    @weight.setter
    def weight(self, value):
        # Update the quantized_weights when the weight property is set
        self.quantized_weights.data = torch.sign(value).to(torch.int8)

    def dequantize_weights(self):
        # Convert quantized_weights back to bfloat16 and compute alpha for the weights
        bfloat16_weights = self.quantized_weights.to(torch.bfloat16)
        alpha = bfloat16_weights.mean()
        return bfloat16_weights * alpha

    def ste_binarize(self, x):
        # Apply the sign function for binarization
        binarized_x = torch.sign(x)
        # Use STE: during backward pass, we bypass the binarization
        binarized_x = (binarized_x - x).detach() + x
        return binarized_x

    def binarize_weights_groupwise(self):
        # Dequantize the weights before binarization
        weights = self.dequantize_weights()

        # Divide weights into groups
        group_size = weights.shape[0] // self.num_groups
        binarized_weights = torch.zeros_like(weights)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            weight_group = weights[start_idx:end_idx]

            # Binarize each group using STE
            alpha_g = weight_group.mean()
            binarized_weights[start_idx:end_idx] = self.ste_binarize(
                weight_group - alpha_g
            )

        return binarized_weights

    def quantize_activations_groupwise(self, x, b=8):
        Q_b = 2 ** (b - 1)

        # Divide activations into groups
        group_size = x.shape[0] // self.num_groups
        quantized_x = torch.zeros_like(x)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            activation_group = x[start_idx:end_idx]

            # Quantize each group
            gamma_g = activation_group.abs().max()
            quantized_x[start_idx:end_idx] = torch.clamp(
                activation_group * Q_b / (gamma_g + self.eps),
                -Q_b + self.eps,
                Q_b - self.eps,
            )

        return quantized_x

    def forward(self, input):
        # Binarize weights (group-wise) using STE
        binarized_weights = self.binarize_weights_groupwise()

        # Normal linear transformation with binarized weights
        output = torch.nn.functional.linear(input, binarized_weights, self.bias)

        # Quantize activations group-wise
        output = self.quantize_activations_groupwise(output)

        return output


class KAL_Net(nn.Module):
    """ Kolmogorov Arnold Legendre Network (KAL-Net)
    Source: https://github.com/1ssb/torchkan
    Source License: MIT
    arxiv paper: https://arxiv.org/abs/2404.19756
    """
    def __init__(self, kan_in_features, kan_out_features, config=None, method=None, bits=None, bias=True):
        super(KAL_Net, self).__init__()  # Initialize the parent nn.Module class

        if config is None:
            config.kan_poly_order = 3
            config.kan_base_activation = "silu"
            config.kan_middle_layers = []

        # Create a list of hidden layers way that is polymorphic with nn.Linear
        self.layers_hidden = []
        self.layers_hidden.extend([kan_in_features])
        self.layers_hidden.extend(config.kan_middle_layers) # middle_layers should be a python list
        self.layers_hidden.extend([kan_out_features])

        # polynomial_order: Order up to which Legendre polynomials are calculated
        self.polynomial_order = config.kan_poly_order
        # base_activation: Activation function used after each layer's computation
        self.base_activation = activation_dictionary[config.kan_base_activation](config)

        # ParameterList for the base weights of each layer
        self.base_weights = nn.ParameterList()
        # ParameterList for the polynomial weights for Legendre expansion
        self.poly_weights = nn.ParameterList()
        # ModuleList for layer normalization for each layer's output
        self.layer_norms = nn.ModuleList()

        # Initialize network parameters
        for i, (in_features, out_features) in enumerate(zip(self.layers_hidden, self.layers_hidden[1:])):
            # Base weight for linear transformation in each layer
            self.base_weights.append(nn.Parameter(torch.randn(out_features, in_features)))
            # Polynomial weight for handling Legendre polynomial expansions
            self.poly_weights.append(nn.Parameter(torch.randn(out_features, in_features * (self.polynomial_order + 1))))
            # Layer normalization to stabilize learning and outputs
            self.layer_norms.append(nn.LayerNorm(out_features))

        # Initialize weights using Kaiming uniform distribution for better training start
        for weight in self.base_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')
        for weight in self.poly_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')

    @lru_cache(maxsize=128)  # Cache to avoid recomputation of Legendre polynomials
    def compute_legendre_polynomials(self, x, order):
        # Base case polynomials P0 and P1
        P0 = x.new_ones(x.shape)  # P0 = 1 for all x
        if order == 0:
            return P0.unsqueeze(-1)
        P1 = x  # P1 = x
        legendre_polys = [P0, P1]

        # Compute higher order polynomials using recurrence
        for n in range(1, order):
            Pn = ((2.0 * n + 1.0) * x * legendre_polys[-1] - n * legendre_polys[-2]) / (n + 1.0)
            legendre_polys.append(Pn)

        return torch.stack(legendre_polys, dim=-1)

    def forward(self, x):
        x = x.to(self.base_weights[0].device)
        batch_size, seq_len, feature_dim = x.size()

        for base_weight, poly_weight, layer_norm in zip(self.base_weights, self.poly_weights, self.layer_norms):
            base_output = F.linear(self.base_activation(x), base_weight)

            # Normalize x to range [-1, 1] for Legendre polynomial computation
            x_min = x.min(dim=1, keepdim=True)[0]
            x_max = x.max(dim=1, keepdim=True)[0]
            x_range = torch.clamp(x_max - x_min, min=1e-6)  # Avoid division by zero
            x_normalized = 2 * (x - x_min) / x_range - 1
            legendre_basis = self.compute_legendre_polynomials(x_normalized, self.polynomial_order)
            legendre_basis = legendre_basis.view(batch_size * seq_len, -1)  # Flatten for linear layer

            poly_output = F.linear(legendre_basis, poly_weight)
            poly_output = poly_output.view(batch_size, seq_len, -1)  # Reshape back to match base_output

            combined_output = base_output + poly_output

            x = self.base_activation(layer_norm(combined_output))

        return x

linear_dictionary = {
    "linear": WrappedLinear,
    "bitlinear": BitLinear,
    "bitlinear_optimized": BitLinearOptimized,
    "bitlinear_1p58": BitLinear1p58,
    "adaptive_bit_linear": AdaptiveBitLinear,
    "kan": KAL_Net,
    "quantized_linear": QuantizedLinear
}
