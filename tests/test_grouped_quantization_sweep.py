import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


DIRECTORY = Path(__file__).parents[1] / "analysis" / "turboquant_angular_distortion"
sys.path.insert(0, str(DIRECTORY))
SPEC = importlib.util.spec_from_file_location(
    "grouped_quantization_sweep", DIRECTORY / "grouped_quantization_sweep.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class GroupedQuantizationSweepTest(unittest.TestCase):
    def test_group_sizes_cover_16_through_2048_by_powers_of_two(self):
        self.assertEqual(MODULE.GROUP_SIZES,
                         [16, 32, 64, 128, 256, 512, 1024, 2048])

    def test_power_of_two_scale_rounds_up(self):
        actual = MODULE.ceil_power_of_two(np.array([0.3, 1.0, 2.1]))
        np.testing.assert_allclose(actual, [0.5, 1.0, 4.0])

    def test_symmetric_and_asymmetric_group_quantizers_are_finite(self):
        values = np.linspace(-1.3, 2.7, 32)
        for asymmetric in (False, True):
            for power_of_two in (False, True):
                quantized = MODULE.grouped_int_quantize(
                    values, 16, asymmetric=asymmetric,
                    power_of_two_scale=power_of_two)
                self.assertTrue(np.all(np.isfinite(quantized)))

    def test_nvfp4_and_mxint8_native_quantizers_are_finite(self):
        values = np.linspace(-1.0, 1.0, 64)
        self.assertTrue(np.all(np.isfinite(MODULE.nvfp4_quantize(values))))
        self.assertTrue(np.all(np.isfinite(MODULE.mxint_quantize(values, 8))))

    def test_small_sweep_includes_four_methods_and_native_references(self):
        curves, summaries = MODULE.collect_metrics(
            dimension=32, group_sizes=[16, 32], angles=np.array([0.0, 45.0]),
            trials=2, seed=11)
        methods = {row.method for row in summaries}
        self.assertTrue(set(MODULE.METHOD_STYLES).issubset(methods))
        self.assertIn("NVFP4", methods)
        self.assertIn("MXINT8", methods)
        self.assertNotIn("MXINT4", methods)
        self.assertTrue(all(np.isfinite(row.mean_distortion_deg) for row in curves))


if __name__ == "__main__":
    unittest.main()
