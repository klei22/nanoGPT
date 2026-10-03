"""Final attention-residual adapter for SmolLM2-style causal LMs."""

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.nn import functional as F


class FinalAttentionResidual(nn.Module):
    """Use a pseudo-query to attend over a model's hidden-state depth."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if not hidden_states:
            raise ValueError("final attention residual requires hidden states")
        values = torch.stack(tuple(hidden_states), dim=0)
        keys = F.rms_norm(values, (values.size(-1),), eps=self.eps)
        scores = torch.einsum("dbtc,c->dbt", keys, self.query)
        weights = scores.softmax(dim=0)
        mixture = torch.einsum("dbt,dbtc->btc", weights, values)
        return mixture, weights


class SmolLM2FinalAttentionResidual:
    """Patch a SmolLM2/Llama causal LM while preserving its HF generation API.

    Hooks collect the token embeddings and decoder-layer outputs. Immediately
    before the model's final RMSNorm, the usual last-layer residual is replaced
    by their learned depth mixture. Because the original ``PreTrainedModel`` is
    returned unchanged, ``generate`` and lm-evaluation-harness remain usable.
    """

    module_name = "final_attention_residual"

    def __init__(self, model: nn.Module, eps: float = 1e-6):
        decoder = self._decoder(model)
        if hasattr(model, self.module_name):
            raise ValueError("a final attention residual is already installed")
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        hidden_size = model.config.hidden_size
        self.model = model
        self.sources: list[torch.Tensor] = []
        self.last_depth_weights: torch.Tensor | None = None
        embedding_weight = decoder.embed_tokens.weight
        self.residual = FinalAttentionResidual(hidden_size, eps).to(
            device=embedding_weight.device,
            dtype=embedding_weight.dtype,
        )
        model.add_module(self.module_name, self.residual)

        self.handles = [decoder.embed_tokens.register_forward_hook(self._capture_embedding)]
        self.handles.extend(layer.register_forward_hook(self._capture_layer) for layer in decoder.layers)
        self.handles.append(decoder.norm.register_forward_pre_hook(self._mix_before_final_norm))

    @staticmethod
    def _decoder(model: nn.Module) -> nn.Module:
        decoder = getattr(model, "model", None)
        required = ("embed_tokens", "layers", "norm")
        if decoder is None or not all(hasattr(decoder, name) for name in required):
            raise TypeError("expected a SmolLM2/Llama-style model.model decoder")
        return decoder

    def _capture_embedding(self, _module, _inputs, output: torch.Tensor) -> None:
        self.sources = [output]

    def _capture_layer(self, _module, _inputs, output) -> None:
        self.sources.append(output[0] if isinstance(output, tuple) else output)

    def _mix_before_final_norm(self, _module, _inputs) -> tuple[torch.Tensor]:
        mixed, self.last_depth_weights = self.residual(self.sources)
        return (mixed,)

    def save(self, path: str) -> None:
        torch.save(self.residual.state_dict(), path)

    def load(self, path: str, map_location: str | torch.device = "cpu") -> None:
        self.residual.load_state_dict(torch.load(path, map_location=map_location, weights_only=True))

    def remove(self) -> None:
        """Remove hooks and the registered adapter module."""
        for handle in self.handles:
            handle.remove()
        delattr(self.model, self.module_name)

    @property
    def trainable_parameter_names(self) -> list[str]:
        return [name for name, parameter in self.model.named_parameters() if parameter.requires_grad]
