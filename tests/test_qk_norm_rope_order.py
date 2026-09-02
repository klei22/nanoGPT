import torch

from variations.attention_variations import _apply_qk_norm_and_rope


class RecordingRoPE:
    def __init__(self):
        self.input = None

    def __call__(self, tensor):
        self.input = tensor.clone()
        return tensor.roll(1, dims=-1)


def test_qk_norm_can_run_before_or_after_rope():
    q = torch.tensor([[[[3.0, 4.0, 0.0, 0.0]]]])
    k = torch.tensor([[[[0.0, 0.0, 5.0, 12.0]]]])

    q_rope_first = RecordingRoPE()
    k_rope_first = RecordingRoPE()
    _apply_qk_norm_and_rope(
        q, k, q_rope_first, k_rope_first,
        use_qk_norm=True, qk_norm_before_rope=False,
    )
    torch.testing.assert_close(q_rope_first.input, q)
    torch.testing.assert_close(k_rope_first.input, k)

    q_norm_first = RecordingRoPE()
    k_norm_first = RecordingRoPE()
    q_result, k_result = _apply_qk_norm_and_rope(
        q, k, q_norm_first, k_norm_first,
        use_qk_norm=True, qk_norm_before_rope=True,
    )
    torch.testing.assert_close(q_norm_first.input.norm(dim=-1), torch.ones(1, 1, 1))
    torch.testing.assert_close(k_norm_first.input.norm(dim=-1), torch.ones(1, 1, 1))
    torch.testing.assert_close(q_result, q_norm_first.input.roll(1, dims=-1))
    torch.testing.assert_close(k_result, k_norm_first.input.roll(1, dims=-1))
