import torch

from utils.predictive_width_metrics import participation_dimension, jensen_gaps, hessian_branch_correlation


def test_participation_dimension_known_rank():
    x = torch.eye(4).repeat(20, 1)
    assert 2.9 < participation_dimension(x) < 3.1  # centering removes one direction


def test_jensen_quadratic_matches_small_noise():
    torch.manual_seed(2)
    base = torch.randn(3, 5)
    logits = base.unsqueeze(0) + 1e-3 * torch.randn(100, 3, 5)
    gaps = jensen_gaps(logits)
    assert torch.allclose(gaps["exact"], gaps["quadratic"], atol=2e-7, rtol=0.1)


def test_identical_branches_have_one_effective_stream():
    branch = torch.randn(2, 3, 7)
    result = hessian_branch_correlation(torch.stack([branch, branch], dim=-2))
    assert torch.allclose(result["effective_streams"], torch.tensor(1.0), atol=1e-5)
