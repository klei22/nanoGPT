import math

import pytest
import torch
from torch.nn import functional as F

from gpt_conf import GPTConfig
from model import GPT
from variations.small_circle_embeddings import SmallCircleEmbedding


@pytest.fixture(autouse=True)
def small_cpu_work():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("dim", [3, 8])
def test_continuous_curve_stays_on_sphere_and_closes(dim):
    torch.manual_seed(12)
    module = SmallCircleEmbedding(10, dim, radius=2.5).double()
    phase = torch.linspace(-2, 2, 101, dtype=torch.float64)
    points = module.embed_phase(phase)
    assert torch.allclose(points.norm(dim=-1), torch.full((101,), 2.5, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(module.embed_phase(0.137), module.embed_phase(1.137), atol=1e-10)
    g = module.geometry()
    assert torch.allclose(module.weight.mean(0), g["center"], atol=2e-7)
    steps = torch.roll(module.weight, -1, 0) - module.weight
    assert torch.allclose(steps.norm(dim=-1), steps.norm(dim=-1)[0].expand(10), atol=2e-6)
    loss = (points * torch.randn_like(points)).sum()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


def tiny_config(**kwargs):
    values = dict(n_embd=3, n_layer=1, n_head=1, n_kv_group=1,
                  vocab_size=10, vocab_sizes=[10, 5], multicontext=True,
                  block_size=4, use_abs_pos_embeddings=False, dropout=0,
                  multicontext_embedding_variant="small_circle", wte_fixed_norm=True)
    values.update(kwargs)
    return GPTConfig(**values)


def test_tied_heads_share_one_final_norm_and_exact_averaged_ce():
    model = GPT(tiny_config())
    tokens = {"digits": torch.tensor([[0, 1, 2, 3]]), "letters": torch.tensor([[0, 1, 2, 3]])}
    targets = {"digits": tokens["digits"] + 1, "letters": (tokens["letters"] + 1) % 5}
    hidden = []
    inputs = []
    hooks = [model.transformer.ln_f.register_forward_hook(lambda m, i, o: hidden.append(o)),
             model.transformer.h[0].register_forward_pre_hook(lambda m, i: inputs.append(i[0]))]
    logits, losses = model(None, token_dict=tokens, target_dict=targets)
    for hook in hooks:
        hook.remove()
    assert len(hidden) == 1
    assert torch.allclose(inputs[0], sum(model.transformer[f"wte_{i}"](t) for i, t in enumerate(tokens.values())))
    expected_losses = []
    for i, target in enumerate(targets.values()):
        embedding = model.transformer[f"wte_{i}"]
        assert model.transformer[f"lm_head_{i}"] is embedding
        expected = F.linear(hidden[0], embedding.weight)
        assert torch.allclose(logits[i], expected)
        expected_losses.append(F.cross_entropy(expected.reshape(-1, expected.size(-1)), target.reshape(-1)))
    assert torch.allclose(torch.stack(losses).mean(), torch.stack(expected_losses).mean())
    torch.stack(losses).mean().backward()
    for i in range(2):
        assert model.transformer[f"wte_{i}"].raw_frame.grad.abs().sum() > 0
    before = [model.transformer[f"wte_{i}"].weight.detach().clone() for i in range(2)]
    model.reproject_token_embeddings()
    assert all(torch.equal(before[i], model.transformer[f"wte_{i}"].weight) for i in range(2))


def test_float_phases_equal_integer_tokens_and_attention_is_causal():
    model = GPT(tiny_config()).eval()
    tokens = {"digits": torch.tensor([[0, 1, 2, 3]]), "letters": torch.tensor([[0, 1, 2, 3]])}
    phases = {"digits": tokens["digits"] / 10, "letters": tokens["letters"] / 5}
    # Targets request all positions; they do not alter the hidden computation.
    original, _ = model(None, token_dict=tokens, target_dict=tokens)
    continuous, _ = model(None, token_dict=phases, target_dict=tokens)
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(original, continuous))
    modified = {key: value.clone() for key, value in tokens.items()}
    modified["digits"][0, -1] = 9
    changed, _ = model(None, token_dict=modified, target_dict=tokens)
    assert all(torch.allclose(a[:, :-1], b[:, :-1], atol=1e-6) for a, b in zip(original, changed))


@pytest.mark.parametrize("change", [dict(wte_weight_tying=False), dict(multicontext=False),
                                    dict(numerical_multicontext=True), dict(n_embd_wte=2),
                                    dict(quantize_wte=True)])
def test_unsupported_circle_combinations_fail_explicitly(change):
    with pytest.raises(ValueError, match="small_circle requires"):
        GPT(tiny_config(**change))


def test_great_circle_limit_and_constant_center_logit_cancellation():
    module = SmallCircleEmbedding(10, 3, offset_init=0, learn_offset=False)
    assert module.geometry()["center"].count_nonzero() == 0
    other = SmallCircleEmbedding(10, 3, offset_init=0.8, learn_offset=False)
    g = other.geometry()
    hidden = torch.randn(6, 3)
    logits = hidden @ other.weight.T
    centered_logits = hidden @ (other.weight - g["center"]).T
    assert torch.allclose(logits.softmax(-1), centered_logits.softmax(-1), atol=1e-6)
    phases = other.decode_phase(hidden)
    exact = (hidden * other.embed_phase(phases)).sum(-1)
    assert (exact[:, None] >= logits - 1e-6).all()


def test_legacy_table_multicontext_default_is_still_tied():
    model = GPT(tiny_config(multicontext_embedding_variant="table"))
    for i in range(2):
        assert isinstance(model.transformer[f"wte_{i}"], torch.nn.Embedding)
        assert model.transformer[f"wte_{i}"].weight is model.transformer[f"lm_head_{i}"].weight
