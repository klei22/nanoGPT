import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


DIRECTORY = Path(__file__).parents[1] / "analysis" / "turboquant_angular_distortion"
sys.path.insert(0, str(DIRECTORY))
SPEC = importlib.util.spec_from_file_location(
    "high_dim_angle_space_evenness", DIRECTORY / "high_dim_angle_space_evenness.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class HighDimAngleSpaceEvennessTest(unittest.TestCase):
    def test_s2_pair_cosine_cdf_is_uniform(self):
        cosine = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        np.testing.assert_allclose(MODULE.sphere_cosine_cdf(cosine, 3),
                                   (cosine + 1.0) / 2.0)

    def test_streamed_metric_evaluation_is_finite(self):
        row = MODULE.evaluate_format("tq3", dimension=64, samples=128,
                                     batch_size=17, seed=9)
        self.assertEqual(row.dimension, 64)
        self.assertEqual(row.samples, 128)
        self.assertTrue(all(np.isfinite(value) for value in (
            row.mean_cosine, row.cosine_std, row.cosine_ks_discrepancy,
            row.resultant_norm, row.second_moment_anisotropy,
        )))


if __name__ == "__main__":
    unittest.main()
