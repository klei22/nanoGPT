import unittest

import torch

from utils.gradient_noise_diagnostics import component_gradient_diagnostics


class GradientNoiseDiagnosticsTests(unittest.TestCase):
    def test_matches_hand_computed_scalar_gradients(self):
        parameter = torch.nn.Parameter(torch.tensor(0.0))
        parameter.grad = torch.tensor(17.0)
        losses = [parameter, 3.0 * parameter]

        result = component_gradient_diagnostics(losses, [parameter])

        self.assertAlmostEqual(result.mean_squared_norm, 5.0)
        self.assertAlmostEqual(result.squared_mean_norm, 4.0)
        self.assertAlmostEqual(result.noise_variance, 2.0)
        self.assertAlmostEqual(result.coherence, 0.8)
        self.assertAlmostEqual(result.noise_scale, 1.25)
        self.assertEqual(parameter.grad.item(), 17.0)

    def test_rejects_single_component(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        with self.assertRaisesRegex(ValueError, "at least two"):
            component_gradient_diagnostics([parameter.square()], [parameter])


if __name__ == "__main__":
    unittest.main()
