import unittest
from types import SimpleNamespace

import torch

from variations.softmax_variations import ReLU2Max


class TestReLU2Max(unittest.TestCase):
    @staticmethod
    def config(*, use_kernel=False, div_by_seq_len=False, divisor=4.0):
        return SimpleNamespace(
            relu2max_divisor=divisor,
            relu2max_use_kernel=use_kernel,
            div_by_seq_len=div_by_seq_len,
        )

    def test_eager_values_and_sequence_scaling(self):
        inputs = torch.tensor([[[-2.0, 0.0, 2.0, 4.0]]])
        layer = ReLU2Max(self.config(div_by_seq_len=True))
        expected = torch.tensor([[[0.0, 0.0, 0.25, 1.0]]])
        torch.testing.assert_close(layer(inputs), expected)

    def test_kernel_matches_eager_forward_and_backward(self):
        eager_input = torch.randn(2, 3, 5, requires_grad=True)
        kernel_input = eager_input.detach().clone().requires_grad_(True)
        eager = ReLU2Max(self.config(div_by_seq_len=True))(eager_input)
        kernel = ReLU2Max(
            self.config(use_kernel=True, div_by_seq_len=True)
        )(kernel_input)

        torch.testing.assert_close(kernel, eager)
        eager.sum().backward()
        kernel.sum().backward()
        torch.testing.assert_close(kernel_input.grad, eager_input.grad)

    def test_rejects_non_positive_divisor(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            ReLU2Max(self.config(divisor=0.0))


if __name__ == "__main__":
    unittest.main()
