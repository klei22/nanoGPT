import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = (Path(__file__).parents[1] / "analysis" / "vector_distribution" /
               "vector_distribution_analysis.py")
SPEC = importlib.util.spec_from_file_location("vector_distribution_analysis", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TurboQuantFormatTest(unittest.TestCase):
    def test_codebook_sizes_and_symmetry(self):
        for bits in range(1, 5):
            values = MODULE.get_values(f"turboquant{bits}")
            self.assertEqual(len(values), 2 ** bits)
            np.testing.assert_allclose(values, -values[::-1], atol=7e-5)
            self.assertTrue(np.all(np.diff(values) > 0))

    def test_paper_two_bit_centroids(self):
        np.testing.assert_allclose(
            MODULE.get_values("turboquant2"),
            [-1.510017, -0.4526475, 0.4526475, 1.510017],
        )


if __name__ == "__main__":
    unittest.main()
