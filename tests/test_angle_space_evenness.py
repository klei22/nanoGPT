import importlib.util
from pathlib import Path
import sys
import unittest

import healpy as hp
import numpy as np


DIRECTORY = Path(__file__).parents[1] / "analysis" / "turboquant_angular_distortion"
sys.path.insert(0, str(DIRECTORY))
SPEC = importlib.util.spec_from_file_location("angle_space_evenness",
                                              DIRECTORY / "angle_space_evenness.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class AngleSpaceEvennessTest(unittest.TestCase):
    def test_perfectly_uniform_histogram(self):
        histogram = np.ones(48)
        coverage, entropy, effective, js, cv = MODULE.histogram_metrics(histogram)
        self.assertAlmostEqual(coverage, 1.0)
        self.assertAlmostEqual(entropy, 1.0)
        self.assertAlmostEqual(effective, 1.0)
        self.assertAlmostEqual(js, 0.0)
        self.assertAlmostEqual(cv, 0.0)

    def test_healpix_centers_are_detected_as_uniform(self):
        nside = 2
        directions = np.asarray(hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))).T
        pixels = hp.vec2pix(nside, directions[:, 0], directions[:, 1], directions[:, 2])
        histogram = np.bincount(pixels, minlength=hp.nside2npix(nside))
        self.assertEqual(np.count_nonzero(histogram), histogram.size)
        self.assertAlmostEqual(MODULE.histogram_metrics(histogram)[1], 1.0)

    def test_moment_metrics_detect_antipodal_axis_bias(self):
        directions = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        dipole, quadrupole = MODULE.moment_metrics(directions)
        self.assertAlmostEqual(dipole, 0.0)
        self.assertGreater(quadrupole, 1.0)

    def test_both_weightings_report_code_frequency_and_unique_support(self):
        rows = MODULE.evaluate_format("int3", samples=512, nside=2,
                                      cap_count=8, seed=3, weighting="both")
        self.assertEqual([row.weighting for row in rows], ["codes", "unique"])
        self.assertEqual(rows[0].samples, 512)
        self.assertLessEqual(rows[1].samples, rows[0].samples)


if __name__ == "__main__":
    unittest.main()
