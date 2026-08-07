import math

import torch

from gpt_conf import GPTConfig
from variations.attention_variations import CausalSelfAttention, InfiniteHeadAttention


def _config(**overrides):
    values = dict(
        block_size=8,
        n_layer=1,
        n_head=2,
        n_kv_group=2,
        n_embd=8,
        dropout=0.0,
        use_qk_norm=True,
        use_qk_norm_scale=True,
        disable_flash_attention=True,
    )
    values.update(overrides)
    return GPTConfig(**values)


def test_per_channel_qk_scale_has_separate_gains_and_gradients():
    config = _config(use_qk_norm_scale_per_channel=True)
    attention = CausalSelfAttention(config)

    expected_gain = math.sqrt(math.log2(config.block_size**2 - config.block_size))
    torch.testing.assert_close(
        attention.qk_norm_q_gain,
        torch.full((config.n_embd // config.n_head,), expected_gain),
    )
    torch.testing.assert_close(attention.qk_norm_q_gain, attention.qk_norm_k_gain)
    assert attention.qk_norm_q_gain is not attention.qk_norm_k_gain
    assert not hasattr(attention, "qk_norm_factor")

    attention(torch.randn(2, config.block_size, config.n_embd), iter_num=0).sum().backward()
    assert attention.qk_norm_q_gain.grad is not None
    assert attention.qk_norm_k_gain.grad is not None


def test_legacy_qk_scale_remains_a_scalar():
    attention = CausalSelfAttention(_config(use_qk_norm_scale_per_channel=False))

    assert attention.qk_norm_factor.shape == torch.Size([])
    assert not hasattr(attention, "qk_norm_q_gain")
    assert not hasattr(attention, "qk_norm_k_gain")


def test_infinite_attention_uses_configured_qk_channel_dimension():
    config = _config(
        use_qk_norm_scale_per_channel=True,
        n_qk_head_dim=6,
        n_v_head_dim=4,
        n_cproj=1,
    )
    attention = InfiniteHeadAttention(config)

    assert attention.qk_norm_q_gain.shape == (config.n_qk_head_dim,)
    assert attention.qk_norm_k_gain.shape == (config.n_qk_head_dim,)
