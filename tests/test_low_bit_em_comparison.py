import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


DIRECTORY = Path(__file__).parents[1] / "analysis" / "turboquant_angular_distortion"
sys.path.insert(0, str(DIRECTORY))
SPEC = importlib.util.spec_from_file_location(
    "low_bit_em_comparison", DIRECTORY / "low_bit_em_comparison.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class LowBitEMComparisonTest(unittest.TestCase):
    def test_legal_splits_use_one_sign_bit(self):
        self.assertEqual(MODULE.LEGAL_EM_FORMATS[3], [(1, 1), (2, 0)])
        self.assertEqual(MODULE.LEGAL_EM_FORMATS[4], [(1, 2), (2, 1), (3, 0)])
        for total_bits, formats in MODULE.LEGAL_EM_FORMATS.items():
            self.assertTrue(all(1 + exponent + mantissa == total_bits
                                for exponent, mantissa in formats))

    def test_all_finite_codebooks_have_no_nan_or_infinity(self):
        for formats in MODULE.LEGAL_EM_FORMATS.values():
            for exponent, mantissa in formats:
                values = MODULE.all_finite_em_values(exponent, mantissa)
                self.assertTrue(np.all(np.isfinite(values)))
                self.assertIn(0.0, values)

    def test_small_shared_pair_sweep_covers_all_formats(self):
        curves, summaries = MODULE.collect_metrics(
            dimensions=[16], angles=np.array([0.0, 45.0]), trials=2,
            clip_sigma=3.0, seed=5)
        expected = {"INT3", "TQ3", "E1M1", "E2M0",
                    "INT4", "TQ4", "E1M2", "E2M1", "E3M0"}
        self.assertEqual({row.format for row in summaries}, expected)
        self.assertEqual(len(curves), 2 * len(expected))
        self.assertTrue(all(np.isfinite(row.mean_distortion_deg) for row in curves))


if __name__ == "__main__":
    unittest.main()
