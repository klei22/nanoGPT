from types import SimpleNamespace

import pytest
import torch

from variations.softmax_variations import ReLU2MaxLine


def make_config(**overrides):
    values = {
        "relu2max_divisor": 1.0,
        "relu2max_line_transition": 3.0,
        "div_by_seq_len": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_relu2max_line_values_and_linear_tail():
    activation = ReLU2MaxLine(make_config())
    x = torch.tensor([-1.0, 2.0, 3.0, 4.0, 5.0])

    torch.testing.assert_close(
        activation(x), torch.tensor([0.0, 4.0, 9.0, 15.0, 21.0])
    )


def test_relu2max_line_has_smooth_derivative_at_transition():
    activation = ReLU2MaxLine(make_config())
    epsilon = 1e-4
    x = torch.tensor([3.0 - epsilon, 3.0, 3.0 + epsilon], requires_grad=True)
    gradients = torch.autograd.grad(activation(x).sum(), x)[0]

    torch.testing.assert_close(gradients, torch.full_like(gradients, 6.0), atol=3e-4, rtol=0)


@pytest.mark.parametrize("field,value", [("relu2max_divisor", 0.0), ("relu2max_line_transition", -1.0)])
def test_relu2max_line_rejects_invalid_parameters(field, value):
    with pytest.raises(ValueError):
        ReLU2MaxLine(make_config(**{field: value}))
