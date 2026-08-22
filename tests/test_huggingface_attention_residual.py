from types import SimpleNamespace

import torch
import torch.nn as nn

from huggingface_model.attention_residual_finetune import FinalAttentionResidualCausalLM


class TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.embedding = nn.Embedding(7, 4)
        self.block = nn.Linear(4, 4)
        self.head = nn.Linear(4, 7, bias=False)

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids, **kwargs):
        first = self.embedding(input_ids)
        second = self.block(first)
        return SimpleNamespace(hidden_states=(first, second))


def test_only_final_attention_query_trains():
    model = FinalAttentionResidualCausalLM(TinyCausalLM())
    model.train()
    tokens = torch.tensor([[0, 1, 2, 3]])
    output = model(tokens, labels=tokens)

    assert model.trainable_parameter_names() == ["residual.query"]
    assert output["logits"].shape == (1, 4, 7)
    assert output["depth_attention_weights"].shape == (2, 1, 4)
    torch.testing.assert_close(output["depth_attention_weights"], torch.full((2, 1, 4), 0.5))

    output["loss"].backward()
    assert model.residual.query.grad is not None
    assert all(parameter.grad is None for parameter in model.base_model.parameters())
