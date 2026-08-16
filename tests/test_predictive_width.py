import torch
import pytest

from gpt_conf import GPTConfig
from variations.predictive_width_variations import StreamReadout


def test_direct_is_explicit_sum_and_parameter_matched():
    torch.manual_seed(1)
    states = torch.randn(2, 3, 4, 8)
    direct = StreamReadout(4, 8, 11, "direct")
    collapsed = StreamReadout(4, 8, 11, "collapsed")
    collapsed.load_state_dict(direct.state_dict())
    assert torch.allclose(direct(states), direct.path_logits(states).sum(-2) / 2)
    assert sum(p.numel() for p in direct.parameters()) == sum(p.numel() for p in collapsed.parameters())
    direct(states).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in direct.parameters())


def test_config_validation():
    GPTConfig().validate_predictive_width()
    with pytest.raises(ValueError, match="divide cleanly"):
        GPTConfig(predictive_width_variant="direct_streams", predictive_head_mode="direct",
                  predictive_stream_dim=7, predictive_stream_n_head=2).validate_predictive_width()
    with pytest.raises(ValueError, match="more than one"):
        GPTConfig(predictive_width_variant="direct_streams", predictive_head_mode="collapsed",
                  predictive_stream_dim=8, predictive_stream_n_head=2).validate_predictive_width()
