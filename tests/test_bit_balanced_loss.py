import unittest

import torch
import torch.nn as nn

from train_variations.loss_variants import BitBalancedCrossEntropy
from utils.bit_usage import compute_total_bit_usage
from variations.linear_variations import linear_dictionary


class _LinearConfig:
    adaptive_linear_init_bits = 6.0
    adaptive_linear_min_bits = 2.0
    adaptive_linear_max_bits = 8.0
    adaptive_linear_activation_bits = 8.0
    adaptive_linear_quantize_input = True
    bias = True


class TinyBitNet(nn.Module):
    def __init__(self):
        super().__init__()
        config = _LinearConfig()
        self.layer = linear_dictionary["adaptive_bit_linear"](
            4, 3, config=config, bits=config.adaptive_linear_init_bits, bias=True
        )

    def forward(self, x):
        return self.layer(x)


class BitBalancedLossTest(unittest.TestCase):
    def test_bit_usage_matches_layer(self):
        model = TinyBitNet()
        total_bits = compute_total_bit_usage(model)
        expected = (
            model.layer.current_bitwidth()
            * (model.layer.weight.numel() + model.layer.bias.numel())
        )
        self.assertTrue(torch.is_tensor(total_bits))
        self.assertTrue(torch.allclose(total_bits, expected, atol=1e-4))

    def test_bit_balanced_loss_adds_penalty_and_grad(self):
        torch.manual_seed(0)
        model = TinyBitNet()
        loss_fn = BitBalancedCrossEntropy(bit_penalty=1e-3)
        loss_fn.set_model(model)
        inputs = torch.randn(2, 4)
        logits = model(inputs)
        targets = torch.tensor([0, 1])

        ce_only = torch.nn.functional.cross_entropy(logits, targets)
        combined = loss_fn(logits, targets)
        self.assertGreater(combined.item(), ce_only.item())

        model.zero_grad()
        combined.backward()
        self.assertIsNotNone(model.layer.bit_param.grad)


if __name__ == "__main__":
    unittest.main()
