import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from huggingface_model.attention_residual_pretrain import (
    AttentionResidualConfig,
    AttentionResidualForCausalLM,
    _token_blocks,
    make_muon_optimizer,
)


@pytest.mark.parametrize(
    ("variant", "activation"),
    [("standard", "softmax"), ("full", "softmax"), ("full", "relu2max_shift")],
)
def test_residual_models_forward_backward(variant, activation):
    config = AttentionResidualConfig(
        vocab_size=32,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=8,
        residual_variant=variant,
        residual_activation=activation,
    )
    model = AttentionResidualForCausalLM(config)
    tokens = torch.randint(0, config.vocab_size, (2, config.block_size))
    output = model(tokens, labels=tokens)
    assert output.logits.shape == (2, config.block_size, config.vocab_size)
    output.loss.backward()
    assert torch.isfinite(output.loss)


def test_shifted_relu2_router_is_normalized_at_zero_initialization():
    config = AttentionResidualConfig(
        vocab_size=16, block_size=4, n_layer=1, n_head=1, n_embd=4,
        residual_variant="full", residual_activation="relu2max_shift",
    )
    router = AttentionResidualForCausalLM(config).residual_router
    scores = torch.zeros(3, 2, 4)
    weights = router.weights(scores)
    torch.testing.assert_close(weights.sum(0), torch.ones(2, 4))
    torch.testing.assert_close(weights, torch.full_like(weights, 1 / 3))


def test_optimizer_routes_embeddings_head_and_rms_gains_to_adam():
    config = AttentionResidualConfig(vocab_size=16, block_size=4, n_layer=1, n_head=1, n_embd=4)
    model = AttentionResidualForCausalLM(config)
    optimizer = make_muon_optimizer(model, 3e-4, 0.0)
    adam_ids = {id(parameter) for parameter in optimizer.param_groups[0]["params"]}
    muon_ids = {id(parameter) for parameter in optimizer.param_groups[1]["params"]}
    assert id(model.wte.weight) in adam_ids
    assert all(id(module.gain) in adam_ids for module in model.modules() if hasattr(module, "gain"))
    assert id(model.blocks[0].attn.qkv.weight) in muon_ids


def test_token_blocks_drops_tokenizer_columns_before_changing_row_count():
    datasets = pytest.importorskip("datasets")

    class TokenizerWithAttentionMask:
        def __call__(self, texts, add_special_tokens=False):
            del add_special_tokens
            input_ids = [[ord(char) for char in text] for text in texts]
            return {
                "input_ids": input_ids,
                "attention_mask": [[1] * len(row) for row in input_ids],
            }

    source = datasets.Dataset.from_dict({"text": ["abc", "defg"]})
    grouped = _token_blocks(source, TokenizerWithAttentionMask(), block_size=2, text_column="text")

    assert grouped.column_names == ["input_ids", "labels"]
    assert grouped["input_ids"] == [[97, 98], [99, 100], [101, 102]]
    assert grouped["labels"] == grouped["input_ids"]
