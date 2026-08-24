from types import SimpleNamespace

import torch
import torch.nn as nn

from huggingface_model.attention_residual_finetune import SmolLM2FinalAttentionResidual
from huggingface_model.attention_residual_finetune.benchmark import build_harness_model


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(7, 4)
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.norm = nn.RMSNorm(4)

    def forward(self, input_ids):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden)


class TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.model = TinyDecoder()
        self.lm_head = nn.Linear(4, 7, bias=False)

    def forward(self, input_ids):
        return self.lm_head(self.model(input_ids))


def test_only_final_attention_query_trains_and_hooks_model():
    model = TinyCausalLM()
    adapter = SmolLM2FinalAttentionResidual(model)
    tokens = torch.tensor([[0, 1, 2, 3]])
    logits = model(tokens)

    assert adapter.trainable_parameter_names == ["final_attention_residual.query"]
    assert logits.shape == (1, 4, 7)
    assert adapter.last_depth_weights is not None
    assert adapter.last_depth_weights.shape == (3, 1, 4)
    torch.testing.assert_close(adapter.last_depth_weights, torch.full((3, 1, 4), 1 / 3))

    logits.sum().backward()
    assert adapter.residual.query.grad is not None
    frozen_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name != "final_attention_residual.query"
    ]
    assert all(parameter.grad is None for parameter in frozen_parameters)


def test_benchmark_loads_harness_from_model_identifier():
    args = SimpleNamespace(
        model="HuggingFaceTB/SmolLM2-135M-Instruct",
        batch_size="auto",
        device="cpu",
    )
    received = {}

    def fake_hflm(**kwargs):
        received.update(kwargs)
        return object()

    build_harness_model(args, model_factory=fake_hflm)

    assert received == {
        "pretrained": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "tokenizer": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "batch_size": "auto",
        "device": "cpu",
    }
