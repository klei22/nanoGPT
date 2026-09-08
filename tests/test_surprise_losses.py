from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from train_variations.loss_variants import build_loss_function, surprise_loss


def _fixture():
    logits = torch.tensor(
        [[[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]], [[1.0, 0.0, 0.0], [3.0, -1.0, 0.0]]],
        requires_grad=True,
    )
    targets = torch.tensor([[0, 1], [2, -1]])
    nll = F.cross_entropy(logits.reshape(-1, 3), targets.reshape(-1), reduction="none", ignore_index=-1)
    return logits, targets, nll[targets.reshape(-1) != -1]


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("median_nll", lambda x: x.median()),
        ("second_moment_nll", lambda x: x.square().mean()),
        ("rms_surprise", lambda x: x.square().mean().sqrt()),
        ("hybrid_ce_squared", lambda x: (x + 0.125 * x.square()).mean()),
        ("power_surprise", lambda x: x.square().mean().sqrt()),
        ("focal_power", lambda x: ((1 - torch.exp(-x)) ** 2 * x.square()).mean()),
        ("capped_tail", lambda x: (x * (1 + 0.25 * torch.minimum(x, x.new_tensor(4.0)))).mean()),
    ],
)
def test_surprise_loss_tokenwise_formulas(variant, expected):
    logits, targets, nll = _fixture()
    actual = surprise_loss(logits, targets, variant=variant)
    torch.testing.assert_close(actual, expected(nll))
    actual.backward()
    assert torch.isfinite(logits.grad).all()


def test_second_moment_aggregation_axes():
    logits, targets, nll = _fixture()
    sequence_means = torch.stack((nll[:2].mean(), nll[2]))
    position_means = torch.stack((torch.stack((nll[0], nll[2])).mean(), nll[1]))
    torch.testing.assert_close(
        surprise_loss(logits, targets, variant="second_moment_nll", aggregation="sequence"),
        sequence_means.square().mean(),
    )
    torch.testing.assert_close(
        surprise_loss(logits, targets, variant="second_moment_nll", aggregation="position"),
        position_means.square().mean(),
    )
    torch.testing.assert_close(
        surprise_loss(logits, targets, variant="second_moment_nll", aggregation="window", window_size=2),
        sequence_means.square().mean(),
    )


def test_builder_configures_focal_and_ignores_invalid_tokens():
    logits, targets, _ = _fixture()
    args = SimpleNamespace(loss_fn="focal", loss_schedule=None, focal_gamma=1.0, focal_alpha=0.5)
    built = build_loss_function(args)
    expected = surprise_loss(logits, targets, variant="focal", focal_gamma=1.0, focal_alpha=0.5)
    torch.testing.assert_close(built(logits, targets), expected)


def test_all_invalid_batch_has_differentiable_zero_loss():
    logits = torch.randn(2, 3, 5, requires_grad=True)
    targets = torch.full((2, 3), -1)
    loss = surprise_loss(logits, targets)
    loss.backward()
    assert loss.item() == 0.0
    assert torch.count_nonzero(logits.grad) == 0
