"""Hook-based, function-preserving attention over decoder-layer residuals."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def find_decoder_components(model: nn.Module) -> tuple[nn.Module, nn.ModuleList, nn.Module]:
    """Return the decoder, layers, and final norm for common Hugging Face CausalLMs."""
    decoder = getattr(model, "model", None) or getattr(model, "transformer", None)
    if decoder is None:
        raise ValueError("model has no supported decoder at .model or .transformer")
    layers = getattr(decoder, "layers", None) or getattr(decoder, "h", None)
    norm = getattr(decoder, "norm", None) or getattr(decoder, "ln_f", None)
    if not isinstance(layers, nn.ModuleList) or norm is None:
        raise ValueError("decoder must expose a ModuleList of layers and a final norm")
    return decoder, layers, norm


class AttentionResidualPEFT(nn.Module):
    """Trainable, gated depth attention that leaves the base model frozen.

    A destination attends over the embedding output and completed decoder-layer
    outputs.  A zero-initialized gate interpolates from the unmodified latest
    residual, making attachment exactly function-preserving at step zero.
    """

    def __init__(self, hidden_size: int, num_destinations: int, eps: float = 1e-6):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(num_destinations, hidden_size))
        self.gates = nn.Parameter(torch.zeros(num_destinations))
        self.eps = eps
        self._sources: list[torch.Tensor] = []
        self._handles: list[Any] = []

    def mix(self, latest: torch.Tensor, destination: int) -> torch.Tensor:
        if not self._sources:
            return latest
        sources = torch.stack(self._sources, dim=0)
        keys = F.rms_norm(sources.float(), (sources.size(-1),), eps=self.eps)
        query = self.queries[destination].float() / math.sqrt(sources.size(-1))
        weights = torch.einsum("dbtc,c->dbt", keys, query).softmax(dim=0)
        mixture = torch.einsum("dbt,dbtc->btc", weights.to(sources.dtype), sources)
        gate = self.gates[destination].tanh().to(latest.dtype)
        return latest + gate * (mixture - latest)

    def attach(self, model: nn.Module) -> "AttentionResidualPEFT":
        if self._handles:
            raise RuntimeError("adapter is already attached")
        decoder, layers, norm = find_decoder_components(model)
        if len(layers) + 1 != self.queries.size(0):
            raise ValueError("adapter destinations must equal decoder layers plus final norm")

        def reset(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
            self._sources = []
            return args, kwargs

        def before_layer(destination: int):
            def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
                if args:
                    hidden = args[0]
                    if not self._sources:
                        self._sources.append(hidden)
                    return (self.mix(hidden, destination), *args[1:]), kwargs
                hidden = kwargs["hidden_states"]
                if not self._sources:
                    self._sources.append(hidden)
                kwargs["hidden_states"] = self.mix(hidden, destination)
                return args, kwargs
            return hook

        def after_layer(_module: nn.Module, _args: tuple[Any, ...], output: Any):
            hidden = output[0] if isinstance(output, tuple) else output
            self._sources.append(hidden)
            return output

        def before_norm(_module: nn.Module, args: tuple[Any, ...]):
            return (self.mix(args[0], len(layers)), *args[1:])

        self._handles.append(decoder.register_forward_pre_hook(reset, with_kwargs=True))
        for index, layer in enumerate(layers):
            self._handles.append(layer.register_forward_pre_hook(before_layer(index), with_kwargs=True))
            self._handles.append(layer.register_forward_hook(after_layer))
        self._handles.append(norm.register_forward_pre_hook(before_norm))
        return self

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._sources = []

    def save(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output / "attention_residual_adapter.pt")

    def load(self, adapter_dir: str | Path) -> None:
        path = Path(adapter_dir) / "attention_residual_adapter.pt"
        self.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))


def attach_attention_residual_peft(model: nn.Module, adapter_dir: str | None = None) -> AttentionResidualPEFT:
    """Freeze a CausalLM, construct its adapter, and optionally load adapter weights."""
    decoder, layers, _ = find_decoder_components(model)
    config = getattr(model, "config", None)
    hidden_size = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)
    if hidden_size is None:
        raise ValueError("model config has no hidden_size or n_embd")
    for parameter in model.parameters():
        parameter.requires_grad = False
    adapter = AttentionResidualPEFT(hidden_size, len(layers) + 1)
    adapter.to(next(model.parameters()).device)
    if adapter_dir:
        adapter.load(adapter_dir)
    adapter.attach(model)
    return adapter
