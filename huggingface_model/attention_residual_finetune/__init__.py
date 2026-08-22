"""Final depth-attention residual fine-tuning for Hugging Face causal LMs."""

from .model import FinalAttentionResidual, FinalAttentionResidualCausalLM

__all__ = ["FinalAttentionResidual", "FinalAttentionResidualCausalLM"]
