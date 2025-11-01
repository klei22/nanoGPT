# variations/linear_variations.py
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
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


class PKLLinear(nn.Module):
    """Linear layer that represents weights as a + b * scale."""

    def __init__(self, in_features, out_features, config=None, method=None, bits=None, bias=None, scale=None, **kwargs):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        default_scale = math.sqrt(2.0)
        if scale is None and config is not None:
            default_scale = getattr(config, "pkl_linear_scale", default_scale)
        elif scale is not None:
            default_scale = scale

        self.register_buffer("scale", torch.tensor(default_scale, dtype=torch.float32))

        self.linear_a = nn.Linear(in_features, out_features, bias=False)
        self.linear_b = nn.Linear(in_features, out_features, bias=False)
        self.bias = None

    @property
    def weight(self):
        scale = self.scale.to(self.linear_b.weight.dtype)
        return self.linear_a.weight + scale * self.linear_b.weight

    def forward(self, x):
        out_a = self.linear_a(x)
        out_b = self.linear_b(x)
        scale = self.scale.to(out_b.dtype)
        return out_a + scale * out_b


class PKLEmbedding(nn.Module):
    """Embedding layer that represents weights as a + b * scale."""

    def __init__(self, num_embeddings, embedding_dim, scale=None, config=None):
        super().__init__()

        default_scale = math.sqrt(2.0)
        if scale is None and config is not None:
            default_scale = getattr(config, "pkl_wte_scale", default_scale)
        elif scale is not None:
            default_scale = scale

        self.register_buffer("scale", torch.tensor(default_scale, dtype=torch.float32))

        self.embedding_a = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_b = nn.Embedding(num_embeddings, embedding_dim)

    @property
    def weight(self):
        scale = self.scale.to(self.embedding_b.weight.dtype)
        return self.embedding_a.weight + scale * self.embedding_b.weight

    def forward(self, x):
        emb_a = self.embedding_a(x)
        emb_b = self.embedding_b(x)
        scale = self.scale.to(emb_b.dtype)
        return emb_a + scale * emb_b


def wrap_with_flashnorm(linear_cls, config):
    """
    Wraps any linear class with FlashNorm-style deferred RMS normalization.
    Only applies if config.use_flash_norm is True.

    Based on "FlashNorm: fast normalization for LLMs"
    Source: https://arxiv.org/pdf/2407.09577
    Key insight: RMSNorm(x) @ W = (x @ W) / RMS(x) when bias=0
    """
    if not getattr(config, "use_flash_norm", False):
        return linear_cls
    
    class FlashNormWrapper(nn.Module):
        def __init__(self, in_features, out_features, config=None, method=None, bits=None, bias=True, **kwargs):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            
            # RMSNorm gain parameter
            self.gain = nn.Parameter(torch.ones(in_features))
            
            # Instantiate the base linear (QuantizedLinear, BitLinear, etc.)
            self.linear = linear_cls(in_features, out_features, config=config, method=method, bits=bits, bias=bias, **kwargs)
            
            # Fuse gain into weights
            self._fuse_gain_into_weights()
        
        def _fuse_gain_into_weights(self):
            """Merge gain into weight matrix: W* = W @ diag(gain)"""
            gain = self.gain.unsqueeze(0)
            if hasattr(self.linear, 'linear_a') and hasattr(self.linear, 'linear_b'):
                with torch.no_grad():
                    self.linear.linear_a.weight.data = self.linear.linear_a.weight.data * gain
                    self.linear.linear_b.weight.data = self.linear.linear_b.weight.data * gain
            elif hasattr(self.linear, 'weight') and self.linear.weight is not None:
                with torch.no_grad():
                    # Broadcast multiply: each output row scaled by corresponding gain
                    self.linear.weight.data = self.linear.weight.data * gain
        
        def forward(self, x):
            rms = x.norm(2, dim=-1, keepdim=True) / math.sqrt(x.size(-1))
            
            # Forward through base linear (gain already fused into weights)
            out = self.linear(x)
            
            # Deferred normalization (happens after matmul)
            out = out / rms
            
            return out
    
    FlashNormWrapper.__name__ = f"FlashNorm_{linear_cls.__name__}"
    return FlashNormWrapper

linear_dictionary = {
    "linear": WrappedLinear,
    "bitlinear": BitLinear,
    "bitlinear_optimized": BitLinearOptimized,
    "bitlinear_1p58": BitLinear1p58,
    "kan": KAL_Net,
    "quantized_linear": QuantizedLinear,
    "pkl_linear": PKLLinear,
}
