import pytest

torch = pytest.importorskip("torch")

from gpt_conf import GPTConfig
from model import GPT


def test_ngpt_forward_backward_and_retraction():
    config = GPTConfig(
        vocab_size=32,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_kv_group=2,
        n_embd=16,
        use_abs_pos_embeddings=False,
        use_ngpt=True,
    )
    model = GPT(config)
    tokens = torch.randint(0, config.vocab_size, (2, config.block_size))
    logits, loss = model(tokens, tokens)
    assert logits.shape == (2, config.block_size, config.vocab_size)
    loss.backward()

    block = model.transformer.h[0]
    assert block.attn_alpha_param.shape == (config.n_embd,)
    assert block.mlp_alpha_param.shape == (config.n_embd,)
    assert block.mlp.ngpt_su.shape == (config.mlp_expansion_factor * config.n_embd,)

    with torch.no_grad():
        model.transformer.wte.weight.mul_(2)
    model.normalize_ngpt_weights()
    norms = model.transformer.wte.weight.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_ngpt_rejects_factored_embeddings():
    with pytest.raises(ValueError, match="n_embd_wte"):
        GPT(GPTConfig(vocab_size=32, block_size=8, n_embd=16, n_head=2,
                      n_embd_wte=8, use_ngpt=True))
