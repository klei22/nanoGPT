import pytest
import torch

from variations.relu2max_attention import make_relu2max_score_mod


def test_relu2max_score_mod_matches_eager():
    scores = torch.tensor([-2.0, 0.0, 3.0])
    score_mod = make_relu2max_score_mod(6.0)
    actual = torch.stack([score_mod(x, None, None, None, None) for x in scores])
    torch.testing.assert_close(actual, torch.relu(scores).square() / 6.0)


def test_relu2max_score_mod_sequence_scaling():
    score_mod = make_relu2max_score_mod(2.0, sequence_length=4)
    assert score_mod(torch.tensor(4.0), None, None, None, None).item() == 2.0


def test_relu2max_rejects_invalid_divisor():
    with pytest.raises(ValueError, match="greater than zero"):
        make_relu2max_score_mod(0)
