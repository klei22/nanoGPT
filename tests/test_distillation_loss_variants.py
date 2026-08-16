import torch

from train_variations.distillation_loss_variants import (
    centered_logit_mse_loss,
    kl_divergence_loss,
    teacher_forward_kl_t1,
)


def test_centered_logit_mse_is_gauge_invariant_and_fp32():
    torch.manual_seed(7)
    student = torch.randn(2, 3, 11, dtype=torch.bfloat16, requires_grad=True)
    teacher = torch.randn_like(student)
    shifts = torch.randn(2, 3, 1, dtype=torch.bfloat16)

    baseline = centered_logit_mse_loss(student, teacher, None)
    shifted = centered_logit_mse_loss(student + shifts, teacher, None)

    assert baseline.dtype == torch.float32
    torch.testing.assert_close(baseline, shifted, atol=2e-3, rtol=2e-3)


def test_centered_logit_mask_and_identity():
    student = torch.randn(1, 3, 5, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([[1, -1, 2]])
    teacher = student.detach().clone()
    teacher[:, 1] += 100

    loss = centered_logit_mse_loss(student, teacher, targets)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.count_nonzero(student.grad) == 0


def test_forward_kl_evaluator_is_fixed_at_temperature_one():
    student = torch.tensor([[[1.0, -1.0, 0.0]]])
    teacher = torch.tensor([[[0.0, 0.5, -0.5]]])
    expected = kl_divergence_loss(student, teacher, None, temperature=1.0)
    torch.testing.assert_close(teacher_forward_kl_t1(student, teacher, None), expected)


def test_distillation_losses_return_fp32_under_mixed_precision():
    student = torch.randn(2, 2, 7, dtype=torch.bfloat16)
    teacher = torch.randn_like(student)
    assert kl_divergence_loss(student, teacher, None).dtype == torch.float32
