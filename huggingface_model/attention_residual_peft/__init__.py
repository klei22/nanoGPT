"""Parameter-efficient depth-wise attention residual adapters."""

from .adapter import AttentionResidualPEFT, find_decoder_components

__all__ = ["AttentionResidualPEFT", "find_decoder_components"]
