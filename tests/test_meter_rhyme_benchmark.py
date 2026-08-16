import unittest

from benchmarks.evaluate_meter_rhyme import macro_f1, normalized_edit_distance, summarize


class MeterRhymeMetricsTests(unittest.TestCase):
    def test_edit_distance(self):
        self.assertEqual(normalized_edit_distance("0101", "0101"), 0.0)
        self.assertEqual(normalized_edit_distance("0101", "0111"), 0.25)

    def test_macro_f1(self):
        self.assertAlmostEqual(macro_f1(["iamb", "iamb", "trochee"], ["iamb", "trochee", "trochee"]), 2 / 3)

    def test_summary_is_clustered_and_per_source(self):
        rows = [{"group_id": "p1", "task": "meter_minimal_pair", "source": "haider", "correct": True,
                 "margin": 1.0, "gold_meter": "iamb", "pred_meter": "iamb",
                 "gold_scansion": "01", "pred_scansion": "01", "meter_success": None, "rhyme_success": None}]
        metrics = summarize(rows)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["per_source"]["haider"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
