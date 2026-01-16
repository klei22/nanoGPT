"""Sparse autoencoder utilities for analyzing transformer activations.

This module defines a lightweight sparse autoencoder that can be trained
on hidden activations extracted from GPT checkpoints. The model exposes
encode/decode helpers and a convenience loss helper for reconstruction
with an L1 sparsity penalty.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SparseAutoencoderConfig:
    """Configuration for :class:`SparseAutoencoder`.

    Attributes:
        input_dim: Dimensionality of the source activations.
        hidden_dim: Size of the latent representation.
        dropout: Dropout probability applied to the latent vector.
        activation: Non-linearity used between encoder and decoder.
        l1_alpha: Weight applied to the L1 sparsity term.
    """

    input_dim: int
    hidden_dim: int
    dropout: float = 0.0
    activation: str = "relu"
    l1_alpha: float = 1e-3


def _get_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    name = name.lower()
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name == "silu":
        return F.silu
    raise ValueError(f"Unsupported activation '{name}'. Choose from relu/gelu/silu.")


class SparseAutoencoder(nn.Module):
    """Two-layer sparse autoencoder with an L1 latent penalty."""

    def __init__(self, config: SparseAutoencoderConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(config.input_dim, config.hidden_dim)
        self.decoder = nn.Linear(config.hidden_dim, config.input_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.activation_fn = _get_activation(config.activation)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent vector prior to sparsity regularization."""
        latent = self.encoder(x)
        latent = self.activation_fn(latent)
        latent = self.dropout(latent)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Reconstruct the original activations from the latent space."""
        return self.decoder(latent)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, latent

    def loss(self, recon: torch.Tensor, latent: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction + L1 sparsity loss."""
        recon_loss = F.mse_loss(recon, target)
        sparsity = latent.abs().mean()
        return recon_loss + self.config.l1_alpha * sparsity
