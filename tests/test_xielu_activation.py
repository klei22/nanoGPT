from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from variations.activation_variations import activation_dictionary


def make_config(**overrides):
    values = {
        "xielu_alpha_p_init": 0.8,
        "xielu_alpha_n_init": 0.8,
        "xielu_beta": 0.5,
        "xielu_eps": 1e-6,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_xielu_matches_reference_equation_and_trains_coefficients():
    activation = activation_dictionary["xielu"](make_config())
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)

    expected = torch.where(
        x > 0,
        0.8 * x.square() + 0.5 * x,
        0.8 * torch.expm1(torch.clamp_max(x, -1e-6)) - 0.8 * x + 0.5 * x,
    )
    torch.testing.assert_close(activation(x), expected)

    activation(x).sum().backward()
    assert activation.alpha_p.grad is not None
    assert activation.alpha_n.grad is not None
    assert F.softplus(activation.alpha_p) > 0
    assert 0.5 + F.softplus(activation.alpha_n) > 0.5


def test_xielu_unused_branch_does_not_create_nan_gradients():
    activation = activation_dictionary["xielu"](make_config()).half()
    # Squaring this value overflows float16. The negative xIELU branch itself
    # remains representable, so the inactive positive branch must not poison it.
    x = torch.tensor([-300.0], dtype=torch.float16, requires_grad=True)

    output = activation(x)
    output.sum().backward()

    assert torch.isfinite(output).all()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(activation.alpha_p.grad).all()
    assert torch.isfinite(activation.alpha_n.grad).all()


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"xielu_alpha_p_init": 0.0}, "alpha_p"),
        ({"xielu_alpha_n_init": 0.5}, "alpha_n"),
        ({"xielu_eps": 0.0}, "eps"),
    ],
)
def test_xielu_rejects_invalid_constraints(override, message):
    with pytest.raises(ValueError, match=message):
        activation_dictionary["xielu"](make_config(**override))
