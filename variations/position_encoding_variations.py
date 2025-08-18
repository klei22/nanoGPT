# variations/position_encoding_variations.py
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from quantization.quantize import dequantize, quantize_dictionary

class QuantizedEmbedding(nn.Embedding):
    def __init__(self, embd_size, embd_dim, quantization_method, quantization_bits):
        super().__init__(embd_size, embd_dim)
        self.quantization_method = quantization_method
        self.quantization_bits = quantization_bits

    def forward(self, x):
        zero_point, weight_norm, quantized_weight = quantize_dictionary[self.quantization_method](self.weight, self.quantization_bits)
        weight = dequantize(zero_point, weight_norm, quantized_weight)
        out = F.embedding(x, weight)
        return out

class RotaryEmbedding(nn.Module):
    """ Implementation of standard Rotary Position Embeddings (RoPE).

    This implementation follows the standard approach for applying RoPE,
    which applies a rotational transformation to each pair of elements in a vector.
    """
    def __init__(self, config=None, size=None):
        super().__init__()
        self.dim = size
        assert self.dim % 2 == 0, "Target length dim must be even for rotary embeddings"
        self.inv_freq = None
        self.start_index = 0
        self.first_pass = True

        self.rope_length = self.dim
        # If rope lenth is set, then trim to rope length
        if config.rope_length:
            assert config.rope_length % 2 == 0, "Rotary length must be even"
            assert config.rope_length <= self.dim, "Rotary length less than or equal to dim"
            self.rope_length = config.rope_length

    def reset_start_index(self):
        """Reset start index to zero."""
        self.start_index = 0

    def increment_start_index(self):
        """Reset start index to zero."""
        self.start_index += 1

    def _generate_inv_freq(self, device):
        """Generate inverse frequencies for RoPE."""
        half_dim = self.rope_length // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, device=device).float() / half_dim))
        return inv_freq

    def update_rope_length(self, rope_length):
        """Update the number of embeddings to do rotations over"""
        if self.rope_length != rope_length:
            assert rope_length % 2 == 0, "New rotary length must be even"
            assert rope_length <= self.dim, "New rotary length must less than or equal to embedding dim"
            self.rope_length = rope_length

    def forward(self, x):
        if self.first_pass:
            self.inv_freq = self._generate_inv_freq(x.device)
            self.first_pass = False

        seq_len = x.shape[-2]
        device = x.device

        # Create position indices
        pos_indices = torch.arange(self.start_index, self.start_index + seq_len, device=x.device).type_as(self.inv_freq)

        # Compute the sinusoidal angles
        angles = torch.einsum('i,d->id', pos_indices, self.inv_freq)
        sin_angles = angles.sin()
        cos_angles = angles.cos()

        # Apply RoPE
        x_even, x_odd = x[..., :self.rope_length:2], x[..., 1:self.rope_length:2]
        x_rotated_even = x_even * cos_angles - x_odd * sin_angles
        x_rotated_odd = x_even * sin_angles + x_odd * cos_angles

        # Reassemble rotated components
        x_combined = torch.empty_like(x, device=device)
        x_combined[..., :self.rope_length:2], x_combined[..., 1:self.rope_length:2] = x_rotated_even, x_rotated_odd

        # Keep the rest of the elements unchanged
        if self.rope_length < self.dim:
            x_combined[..., self.rope_length:] = x[..., self.rope_length:]

        return x_combined


class FourierPositionEmbedding(RotaryEmbedding):
    """Fourier Position Embedding (FoPE).

    Extends RoPE by mixing multiple frequency components in each head using a
    Fourier series and by removing inadequately trained low-frequency terms.
    """

    def __init__(self, config=None, size=None):
        super().__init__(config, size)
        self.config = config
        self.head_dim = size
        self.num_heads = None
        self.input_dim = None
        self.output_dim = None
        self.sin_coef = None
        self.cos_coef = None
        # Allow tweaking of the coefficient initialisation gain
        self.fourier_gain = getattr(config, "rope_fourier_init_norm_gain", 1.0)

    def _generate_inv_freq(self, device):
        inv_freq = super()._generate_inv_freq(device)
        floor = 2 * math.pi / self.config.block_size
        mask = inv_freq >= floor
        return inv_freq[mask]

    def rotate_half(self, x):
        x1 = x[..., : x.size(-1) // 2]
        x2 = x[..., x.size(-1) // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_embed(self, pos_sin, pos_cos, t):
        fourier_sin = torch.einsum(
            "bhtD,hDd->bhtd",
            pos_sin,
            self.sin_coef / self.sin_coef.sum(dim=-2, keepdim=True),
        )
        fourier_cos = torch.einsum(
            "bhtD,hDd->bhtd",
            pos_cos,
            self.cos_coef / self.cos_coef.sum(dim=-2, keepdim=True),
        )
        half_dim = t.size(-1) // 2
        fourier_sin = F.pad(
            fourier_sin, (0, half_dim - fourier_sin.size(-1)), value=1
        )
        fourier_cos = F.pad(
            fourier_cos, (0, half_dim - fourier_cos.size(-1)), value=1
        )
        fourier_sin = torch.cat((fourier_sin, fourier_sin), dim=-1)
        fourier_cos = torch.cat((fourier_cos, fourier_cos), dim=-1)
        return (t * fourier_cos) - (self.rotate_half(t) * fourier_sin)

    def forward(self, x):
        if self.first_pass:
            self.inv_freq = self._generate_inv_freq(x.device)
            self.input_dim = self.inv_freq.size(-1)
            self.output_dim = min(self.input_dim, self.head_dim // 4)
            self.num_heads = x.shape[1]
            self.sin_coef = nn.Parameter(
                torch.randn(self.num_heads, self.input_dim, self.output_dim, device=x.device),
                requires_grad=False,
            )
            self.cos_coef = nn.Parameter(
                torch.randn(self.num_heads, self.input_dim, self.output_dim, device=x.device),
                requires_grad=False,
            )
            nn.init.xavier_normal_(self.sin_coef, gain=self.fourier_gain)
            nn.init.xavier_normal_(self.cos_coef, gain=self.fourier_gain)
            eye = torch.eye(self.input_dim, device=x.device)
            self.sin_coef.data += eye[:, : self.output_dim]
            self.cos_coef.data += eye[:, : self.output_dim]
            self.first_pass = False

        seq_len = x.shape[-2]
        pos_indices = torch.arange(
            self.start_index, self.start_index + seq_len, device=x.device
        ).type_as(self.inv_freq)
        angles = torch.einsum("i,d->id", pos_indices, self.inv_freq)
        pos_sin = angles.sin().unsqueeze(0).unsqueeze(0)
        pos_cos = angles.cos().unsqueeze(0).unsqueeze(0)
        b, h, t, _ = x.size()
        pos_sin = pos_sin.expand(b, h, t, -1)
        pos_cos = pos_cos.expand(b, h, t, -1)
        return self.apply_rotary_pos_embed(pos_sin, pos_cos, x).to(x.dtype)

class SymmetricalOverlapAngularPositions(nn.Module):
    """ SOAP is a fresh and 'clean' implementation of Rotary Embeddings.

    Symmetries and rotational overlap to optimize storage requirements.

    Using symmetries reduces the cache size in specialized hardware
    by a factor of 8. (x <-> y symmetries, x-axis symmetries, y-axis symmetries)

    Applies the same rotation to each pair of elements per vector.

    Likely to generalize with minimal fine-tuning via interpolation.
    """
    def __init__(self, config, size=None, num_angles=None):
        super().__init__()

        self.dim = size
        assert self.dim % 2 == 0, "Target length dim must be even for rotary embeddings"

        self.num_angles = num_angles
        if config.rope_length != None:
            assert config.rope_length % 2 == 0, "Rotary length must be even"
            assert config.rope_length <= self.dim, "Rotary length less than or equal to dim"
            self.rope_length = config.rope_length
        else:
            self.rope_length = self.dim
            assert self.rope_length % 2 == 0, "Embedding dim length must be even for rotary position embeddings"
        self.first_pass = True
        self.angles = None

    def _generate_angles(self, num_angles, device):
        """Generate angles from 0 to 2*pi based on the number of angles."""
        return torch.linspace(0, 2 * math.pi - (2 * math.pi / num_angles), steps=num_angles, device=device)

    def update_num_angles(self, num_angles, device):
        """Update the number of angles and regenerate angles tensor if needed."""
        if self.num_angles != num_angles:
            self.num_angles = num_angles
            self.angles = self._generate_angles(num_angles, device)

    def update_rope_length(self, rope_length):
        """Update the number of embeddings to do rotations over"""
        if self.rope_length != rope_length:
            assert rope_length % 2 == 0, "New rotary length must be even"
            assert rope_length <= self.dim, "New rotary length must less than or equal to embedding dim"
            self.rope_length = rope_length

    def forward(self, x):
        if self.first_pass:
            self.angles = self._generate_angles(self.num_angles, x.device)
            self.first_pass = False

        seq_len = x.shape[-2]
        device = x.device

        self.angles = torch.roll(self.angles, shifts=1, dims=0)

        # Create index list, wrap around as necessary
        angle_indices = torch.arange(seq_len, device=device) % self.num_angles

        # Assign angles
        selected_angles = self.angles[angle_indices]

        # Run angles through sine and cosine
        sin_angles = selected_angles.sin().unsqueeze(-1).repeat(1, self.rope_length // 2)
        cos_angles = selected_angles.cos().unsqueeze(-1).repeat(1, self.rope_length // 2)

        # Split input tensor into even and odd components for the rope length
        x_even, x_odd = x[..., :self.rope_length:2], x[..., 1:self.rope_length:2]

        # Rotate components for the rope length
        x_rotated_even = x_even * cos_angles - x_odd * sin_angles
        x_rotated_odd = x_even * sin_angles + x_odd * cos_angles

        # Reassemble rotated components
        x_combined = torch.empty_like(x, device=device)
        x_combined[..., :self.rope_length:2], x_combined[..., 1:self.rope_length:2] = x_rotated_even, x_rotated_odd

        # Keep the rest of the elements unchanged
        if self.rope_length < self.dim:
            x_combined[..., self.rope_length:] = x[..., self.rope_length:]

        return x_combined


class FIRE(nn.Module):
    """
    _F_unctional _I_nterpolation For _R_elative Position _E_ncoding

    Description:
    An advanced positional embedding method which promises high extrapolation of
    learned context length.  Train with short context and extend e.g. from 2048
    to 8192.

    Arxiv Paper Source: https://arxiv.org/pdf/2310.04418.pdf
    """

    def __init__(self, config, num_heads=12, eps=1e-6):
        super(FIRE, self).__init__()

        mlp_layers = []

        if config.fire_num_hidden_layers >= 1:
            # First linear layer
            mlp_layers.append(nn.Linear(1, config.fire_mlp_width))

            for _ in range(config.fire_num_hidden_layers - 1):
                mlp_layers.append(nn.ReLU())
                mlp_layers.append(nn.Linear(config.fire_mlp_width, config.fire_mlp_width))

            mlp_layers.append(nn.ReLU())
            # Final linear layer
            mlp_layers.append(nn.Linear(config.fire_mlp_width, num_heads))

        elif config.fire_num_hidden_layers == 0:
            mlp_layers.append(nn.Linear(1, num_heads))

        if config.fire_outermost_sigma:
            mlp_layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*mlp_layers)
        self.c = nn.Parameter(torch.tensor(config.fire_init_c, dtype=torch.float))
        self.init_L = nn.Parameter(torch.tensor(config.fire_init_L, dtype=torch.float), requires_grad=False)
        self.L_multiplier = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.eps = eps
        self.fire_log_bias = config.fire_log_bias

    def forward(self, x: torch.Tensor):
        seq_length = x.size(1)
        positions = torch.arange(seq_length, dtype=torch.float, device=x.device)
        rel_distance = positions[:, None] - positions[None, :]

        # Apply absolute value and ensure positive before log
        abs_rel_distance = torch.abs(rel_distance) + self.eps

        threshold = torch.abs(self.L_multiplier * self.init_L)
        pos_normalizer = torch.max(positions, threshold)
        pos_normalizer = pos_normalizer[:, None] + self.eps  # Ensure pos_normalizer is never zero

        # Use safe log operation
        log_rel_distance = torch.log(abs_rel_distance * self.c + self.fire_log_bias + self.eps)
        log_pos_normalizer = torch.log(torch.abs(self.c * pos_normalizer) + self.fire_log_bias + self.eps)

        normalized_distance = log_rel_distance / log_pos_normalizer

        fire_bias = self.mlp(normalized_distance.unsqueeze(-1))
        fire_bias = fire_bias.unsqueeze(0).permute(0, 3, 1, 2)

        return fire_bias

