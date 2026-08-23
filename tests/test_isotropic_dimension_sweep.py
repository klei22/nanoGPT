import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


DIRECTORY = Path(__file__).parents[1] / "analysis" / "turboquant_angular_distortion"
sys.path.insert(0, str(DIRECTORY))
SPEC = importlib.util.spec_from_file_location(
    "isotropic_dimension_sweep", DIRECTORY / "isotropic_dimension_sweep.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class IsotropicDimensionSweepTest(unittest.TestCase):
    def test_default_dimension_range_uses_every_power_of_two(self):
        self.assertEqual(MODULE.power_of_two_dimensions(2, 1024),
                         [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])

    def test_range_selects_only_contained_powers(self):
        self.assertEqual(MODULE.power_of_two_dimensions(5, 33), [8, 16, 32])

    def test_small_sweep_returns_finite_curve_and_summary_metrics(self):
        curves, summaries = MODULE.collect_metrics(
            dimensions=[2, 4], angles=np.array([0.0, 45.0]), bits_list=[3],
            trials=2, clip_sigma=3.0, seed=7)
        self.assertEqual(len(curves), 8)
        self.assertEqual(len(summaries), 4)
        self.assertEqual({row.format for row in summaries}, {"INT3", "TQ3"})
        self.assertTrue(all(np.isfinite(row.mean_distortion_deg) for row in curves))
        self.assertTrue(all(np.isfinite(row.rms_distortion_deg) for row in summaries))


if __name__ == "__main__":
    unittest.main()
