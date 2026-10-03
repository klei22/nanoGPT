import pytest
import torch

from gpt_conf import GPTConfig
from variations.attention_variations import CausalSelfAttention, GemmaRMSNorm


def test_gemma_rms_norm_has_identity_gain_at_initialization():
    norm = GemmaRMSNorm(3)
    x = torch.tensor([[[[1.0, 2.0, 3.0]]]])

    expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

    assert torch.equal(norm.weight, torch.zeros(3))
    assert torch.allclose(norm(x), expected)


def test_gemma_qk_norm_uses_independent_per_channel_gains():
    config = GPTConfig(n_embd=8, n_head=2, use_gemma_qk_norm=True)
    attention = CausalSelfAttention(config)

    assert attention.q_norm.weight.shape == (4,)
    assert attention.k_norm.weight.shape == (4,)
    assert attention.q_norm.weight is not attention.k_norm.weight


def test_qk_norm_variations_are_mutually_exclusive():
    config = GPTConfig(
        n_embd=8,
        n_head=2,
        use_qk_norm=True,
        use_gemma_qk_norm=True,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        CausalSelfAttention(config)
