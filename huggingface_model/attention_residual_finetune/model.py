"""Add a trainable final attention-residual mixer to a frozen causal LM."""

from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F


class FinalAttentionResidual(nn.Module):
    """Use a pseudo-query to attend over a model's hidden-state depth."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        if not hidden_states:
            raise ValueError("final attention residual requires hidden states")
        values = torch.stack(hidden_states, dim=0)
        keys = F.rms_norm(values, (values.size(-1),), eps=self.eps)
        scores = torch.einsum("dbtc,c->dbt", keys, self.query)
        weights = scores.softmax(dim=0)
        mixture = torch.einsum("dbt,dbtc->btc", weights, values)
        return mixture, weights


class FinalAttentionResidualCausalLM(nn.Module):
    """Freeze a Hugging Face causal LM and train only its final depth mixer.

    The wrapped model must accept ``output_hidden_states`` and expose
    ``get_output_embeddings()``, as ``AutoModelForCausalLM`` models do.
    """

    def __init__(self, base_model: nn.Module, eps: float = 1e-6):
        super().__init__()
        self.base_model = base_model
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)

        hidden_size = getattr(base_model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(base_model.config, "n_embd", None)
        if hidden_size is None:
            raise ValueError("base model config must define hidden_size or n_embd")
        if base_model.get_output_embeddings() is None:
            raise ValueError("base model must expose output embeddings")

        self.residual = FinalAttentionResidual(hidden_size, eps)
        self.config = base_model.config

    def train(self, mode: bool = True) -> "FinalAttentionResidualCausalLM":
        super().train(mode)
        # Frozen dropout must not inject noise into the residual sources.
        self.base_model.eval()
        self.residual.train(mode)
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("return_dict", None)
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )

        mixed, depth_weights = self.residual(tuple(outputs.hidden_states))
        logits = self.base_model.get_output_embeddings()(mixed)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
                labels[..., 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return {
            "loss": loss,
            "logits": logits,
            "depth_attention_weights": depth_weights,
            "hidden_states": outputs.hidden_states,
        }

    def trainable_parameter_names(self) -> list[str]:
        """Return an auditable list of parameters updated by the optimizer."""
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]
