import importlib.util
import unittest
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[1] / "finetune.py"
SPEC = importlib.util.spec_from_file_location("rmsnorm_channel_zeroing", SCRIPT)
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


class ToyRMSNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.5, -0.1, 0.3, 0.01]))

    def forward(self, inputs):
        return inputs * self.weight


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = ToyRMSNorm()
        self.linear = torch.nn.Linear(4, 4)


class ChannelZeroingTests(unittest.TestCase):
    def setUp(self):
        self.model = ToyModel()
        self.weights = experiment.find_rmsnorm_weights(self.model)

    def test_selects_lowest_magnitude_per_layer(self):
        selection = experiment.select_smallest_channels(self.weights, 0.5)
        self.assertEqual(selection["norm.weight"].tolist(), [False, True, False, True])

    def test_masks_gradients_and_restores_unselected_values(self):
        selection = experiment.select_smallest_channels(self.weights, 0.25)
        initial = {name: value.detach().clone() for name, value in self.weights.items()}
        handles = experiment.configure_selected_gradients(self.model, self.weights, selection)
        self.model.norm.weight.sum().backward()
        self.assertEqual(self.model.norm.weight.grad.tolist(), [0.0, 0.0, 0.0, 1.0])
        self.model.norm.weight.data.add_(1)
        experiment.restore_unselected(self.weights, initial, selection)
        self.assertTrue(torch.equal(self.model.norm.weight[:3], initial["norm.weight"][:3]))
        self.assertEqual(float(self.model.norm.weight[3]), float(initial["norm.weight"][3] + 1))
        for handle in handles:
            handle.remove()

    def test_threshold_only_zeros_selected_small_values(self):
        selection = experiment.select_smallest_channels(self.weights, 0.5)
        count = experiment.threshold_selected(self.weights, selection, 0.1)
        self.assertEqual(count, 2)
        self.assertEqual(self.model.norm.weight.tolist(), [0.5, 0.0, 0.3, 0.0])


if __name__ == "__main__":
    unittest.main()
