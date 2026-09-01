import unittest

import torch
from torch import nn

from train_variations.confidence_rethinking import confidence_rethinking_forward


class ScriptedModel(nn.Module):
    def __init__(self, logits_by_pass):
        super().__init__()
        self.logits_by_pass = logits_by_pass
        self.inputs = []

    def forward(self, inputs, **_kwargs):
        self.inputs.append(inputs.clone())
        return self.logits_by_pass[len(self.inputs) - 1], None


class ConfidenceRethinkingTest(unittest.TestCase):
    def test_low_confidence_positions_emit_thinking_and_retry(self):
        inputs = torch.tensor([[1, 2]])
        targets = torch.tensor([[1, 1]])
        first = torch.tensor([[[0.0, 4.0, 0.0], [4.0, 0.0, 0.0]]])
        second = torch.tensor([[[0.0, 4.0, 0.0], [0.0, 4.0, 0.0]]])
        model = ScriptedModel([first, second])

        logits, loss = confidence_rethinking_forward(
            model, inputs, targets, thinking_token_id=2,
            confidence_threshold=0.70, max_passes=3,
        )

        self.assertEqual(len(model.inputs), 2)
        self.assertTrue(torch.equal(model.inputs[1], torch.tensor([[1, 2]])))
        self.assertIs(logits, second)
        self.assertTrue(torch.isfinite(loss))

    def test_pass_and_threshold_validation(self):
        model = ScriptedModel([])
        inputs = targets = torch.ones((1, 1), dtype=torch.long)
        with self.assertRaises(ValueError):
            confidence_rethinking_forward(model, inputs, targets, thinking_token_id=0, max_passes=0)
        with self.assertRaises(ValueError):
            confidence_rethinking_forward(
                model, inputs, targets, thinking_token_id=0, confidence_threshold=1.1
            )


if __name__ == '__main__':
    unittest.main()
