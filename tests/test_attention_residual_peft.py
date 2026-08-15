import torch
from torch import nn

from huggingface_model.attention_residual_peft.adapter import attach_attention_residual_peft


class Layer(nn.Module):
    def forward(self, hidden_states):
        return (hidden_states + 1,)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Layer(), Layer()])
        self.norm = nn.Identity()

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)[0]
        return self.norm(hidden_states)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Decoder()
        self.config = type("Config", (), {"hidden_size": 4})()
        self.anchor = nn.Parameter(torch.ones(1))

    def forward(self, hidden_states):
        return self.model(hidden_states)


def test_adapter_is_function_preserving_and_only_adapter_is_trainable():
    model = Model()
    inputs = torch.randn(2, 3, 4)
    expected = model(inputs)
    adapter = attach_attention_residual_peft(model)
    actual = model(inputs)

    torch.testing.assert_close(actual, expected)
    assert sum(p.numel() for p in adapter.parameters()) == 15
    assert not any(p.requires_grad for p in model.parameters())
    assert all(p.requires_grad for p in adapter.parameters())


def test_nonzero_gate_changes_depth_mixture_and_backpropagates():
    model = Model()
    adapter = attach_attention_residual_peft(model)
    adapter.gates.data.fill_(0.5)
    output = model(torch.zeros(1, 1, 4))
    output.sum().backward()

    assert not torch.equal(output, torch.full_like(output, 2.0))
    assert adapter.gates.grad is not None
    assert adapter.queries.grad is not None
