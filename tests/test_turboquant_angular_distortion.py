import importlib.util
from pathlib import Path
import unittest

import numpy as np


PATH = (Path(__file__).parents[1] / "analysis" / "turboquant_angular_distortion" /
        "turboquant_angular_distortion.py")
SPEC = importlib.util.spec_from_file_location("turboquant_angular_distortion", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TurboQuantAngularDistortionTest(unittest.TestCase):
    def test_lloyd_max_matches_paper_two_bit_example(self):
        actual = MODULE.gaussian_lloyd_max_codebook(2)
        np.testing.assert_allclose(actual, [-1.5104, -0.4528, 0.4528, 1.5104], atol=5e-4)

    def test_randomized_hadamard_is_orthogonal(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=16)
        y = rng.normal(size=16)
        signs = rng.choice((-1.0, 1.0), size=16)
        hx = MODULE.randomized_hadamard(x, signs)
        hy = MODULE.randomized_hadamard(y, signs)
        self.assertAlmostEqual(float(x @ y), float(hx @ hy), places=11)
        self.assertAlmostEqual(float(np.linalg.norm(x)), float(np.linalg.norm(hx)), places=11)

    def test_transform_spreads_sparse_vector(self):
        x = np.zeros(32)
        x[0] = 1.0
        transformed = MODULE.randomized_hadamard(x, np.ones(32))
        np.testing.assert_allclose(np.abs(transformed), np.full(32, 1 / np.sqrt(32)))


if __name__ == "__main__":
    unittest.main()
