import torch

from gpt_conf import GPTConfig
from model import GPT
from variations.attention_residual_variations import FullAttentionResidual
from variations.softmax_variations import ReLU2Max


def test_zero_queries_start_as_equal_weight_average():
    mixer = FullAttentionResidual(n_destinations=1, n_embd=2)
    sources = [
        torch.tensor([[[1.0, 3.0]]]),
        torch.tensor([[[5.0, 7.0]]]),
    ]

    result = mixer(sources, destination=0)

    torch.testing.assert_close(result, torch.tensor([[[3.0, 5.0]]]))


def test_relu2max_can_weight_full_attention_residuals():
    config = GPTConfig(relu2max_divisor=1.0, div_by_seq_len=False)
    mixer = FullAttentionResidual(
        n_destinations=1,
        n_embd=2,
        weighting=ReLU2Max(config, dim=0),
    )
    with torch.no_grad():
        mixer.queries[0].copy_(torch.tensor([1.0, 0.0]))
    sources = [
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[[-1.0, 0.0]]]),
    ]

    result = mixer(sources, destination=0)

    torch.testing.assert_close(result, torch.tensor([[[2.0, 0.0]]]))


def test_full_attention_residual_model_forward_and_backward():
    config = GPTConfig(
        block_size=4,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_kv_group=2,
        n_embd=8,
        dropout=0.0,
        attention_residual_variant="full",
    )
    model = GPT(config)
    tokens = torch.randint(0, config.vocab_size, (2, config.block_size))
    targets = torch.randint(0, config.vocab_size, (2, config.block_size))

    logits, loss = model(tokens, targets)
    loss.backward()

    assert logits.shape == (2, config.block_size, config.vocab_size)
    assert model.attention_residual.queries.shape == (2 * config.n_layer + 1, config.n_embd)
    assert model.attention_residual.queries.grad is not None


def test_full_attention_residual_model_uses_relu2max_weighting():
    config = GPTConfig(
        block_size=4,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_kv_group=2,
        n_embd=8,
        attention_residual_variant="full",
        attention_residual_weighting="relu2max",
    )

    model = GPT(config)

    assert isinstance(model.attention_residual.weighting, ReLU2Max)
    assert model.attention_residual.weighting.dim == 0
    assert torch.count_nonzero(model.attention_residual.queries) > 0
