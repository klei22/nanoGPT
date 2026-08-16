from types import SimpleNamespace
import torch

from variations.router_variations import NoisyTopKRouter


def test_noisy_router_selects_expert_axis_and_is_seeded():
    config = SimpleNamespace(n_embd=5, n_experts=7, moe_top_k=2, moe_router_scheme="noisy_top_k")
    router = NoisyTopKRouter(config)
    x = torch.randn(2, 3, 5)
    torch.manual_seed(9)
    first = router.route_with_diagnostics(x)
    torch.manual_seed(9)
    second = router.route_with_diagnostics(x)
    assert first.indices.shape == (2, 3, 2)
    assert torch.equal(first.indices, second.indices)
    assert torch.allclose(first.probabilities.sum(-1), torch.ones(2, 3))
    assert first.load.shape == (7,)
