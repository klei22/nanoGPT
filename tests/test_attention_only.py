import ast
from pathlib import Path

import torch

from gpt_conf import GPTConfig
from model import GPT


def _config(**overrides):
    values = dict(
        vocab_size=32,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_kv_group=1,
        n_embd=16,
        attention_only=True,
        use_abs_pos_embeddings=False,
        use_rotary_embeddings=True,
        use_qk_norm=True,
        dropout=0.0,
    )
    values.update(overrides)
    return GPTConfig(**values)


def test_attention_only_model_has_no_mlp_parameters_and_trains():
    model = GPT(_config())

    assert all(block.mlp is None for block in model.transformer.h)
    assert not any(".mlp." in name for name, _ in model.named_parameters())

    tokens = torch.randint(0, 32, (2, 8))
    logits, loss = model(tokens, tokens, iter_num=0)
    assert logits.shape == (2, 8, 32)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.transformer.h[0].attn.c_proj.weight.grad is not None


def test_attention_only_rejects_parallel_mlp():
    try:
        GPT(_config(use_parallel_mlp=True))
    except ValueError as error:
        assert "use_parallel_mlp" in str(error)
    else:
        raise AssertionError("expected incompatible configuration to fail")


def test_attention_only_is_available_from_training_cli():
    tree = ast.parse((Path(__file__).parents[1] / "train_args.py").read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--attention_only"
    ]
    assert len(calls) == 1

    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    assert isinstance(keywords["default"], ast.Constant)
    assert keywords["default"].value is False
    assert isinstance(keywords["action"], ast.Attribute)
    assert keywords["action"].attr == "BooleanOptionalAction"
