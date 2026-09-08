import pytest
import torch

from quantizations.ptq.convrot import group_rotate, regular_hadamard


@pytest.mark.parametrize("order", [4, 16, 64])
def test_regular_hadamard_is_orthogonal_and_regular(order):
    rotation = regular_hadamard(order)
    assert torch.allclose(rotation @ rotation.T, torch.eye(order), atol=1e-6)
    expected_sum = torch.full((order,), 1.0)
    assert torch.allclose(rotation.sum(0).abs(), expected_sum, atol=1e-6)
    assert torch.allclose(rotation.sum(1).abs(), expected_sum, atol=1e-6)


def test_group_rotation_preserves_linear_operation():
    torch.manual_seed(1)
    activations = torch.randn(5, 32)
    weight = torch.randn(7, 32)
    rotation = regular_hadamard(16)
    actual = group_rotate(activations, rotation) @ group_rotate(weight, rotation).T
    assert torch.allclose(actual, activations @ weight.T, atol=1e-5)


def test_regular_hadamard_rejects_unsupported_order():
    with pytest.raises(ValueError, match="power of four"):
        regular_hadamard(8)
