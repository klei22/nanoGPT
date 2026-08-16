import importlib.util
from pathlib import Path
import unittest
import tempfile

import yaml


MODULE_PATH = (
    Path(__file__).parents[1] / "optimization_and_search" / "run_experiments.py"
)
SPEC = importlib.util.spec_from_file_location("run_experiments", MODULE_PATH)
run_experiments = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_experiments)


class SweepDefaultAndNullTests(unittest.TestCase):
    def test_unquoted_default_omits_parameter_and_null_preserves_none(self):
        config = yaml.safe_load(
            'norm_variant_wte: [default, "hyperspherenorm", null]'
        )

        combinations = [
            combo for combo, _common_keys in run_experiments.generate_combinations(config)
        ]

        self.assertEqual(
            combinations,
            [
                {},
                {"norm_variant_wte": "hyperspherenorm"},
                {"norm_variant_wte": None},
            ],
        )

    def test_null_is_not_rendered_as_a_cli_string(self):
        command = run_experiments.build_command(
            {
                "norm_variant_wte": None,
                "activation_variant": "gelu",
            }
        )

        self.assertEqual(command, ["python3", "train.py", "--activation_variant", "gelu"])

    def test_default_in_common_group_is_omitted(self):
        config = {"common_group": {"norm_variant_wte": ["default"]}}

        self.assertEqual(list(run_experiments.generate_combinations(config)), [({}, set())])


class MetricsSchemaTests(unittest.TestCase):
    def _read(self, values):
        with tempfile.TemporaryDirectory() as out_dir:
            path = Path(out_dir) / run_experiments.METRICS_FILENAME
            path.write_text(", ".join(str(value) for value in values) + "\n")
            return run_experiments.read_metrics(out_dir)

    def test_current_schema_preserves_top_level_metrics(self):
        core = [
            1.5, 2.1, 1250, 10000, 12000000, 3.5, 0.25,
            100, 200, 300, 40, "", 0.6, 0.55, 3.2, 0.3, 0.4,
            12, 0.99, 0.4, 0.15, 65, 1.4, 0.29, 1.48, 0.31,
        ]
        metrics = self._read(core + [0.0] * 10)
        self.assertEqual(metrics["best_val_iter"], 1250)
        self.assertEqual(metrics["num_params"], 12000000)
        self.assertEqual(metrics["avg_top1_prob"], 0.6)
        self.assertEqual(metrics["teacher_forward_kl_t1"], 0.31)

    def test_previous_schema_inserts_missing_common_kl_before_stats(self):
        core = [
            1.5, 2.1, 1250, 10000, 12000000, 3.5, 0.25,
            100, 200, 300, 40, "", 0.6, 0.55, 3.2, 0.3, 0.4,
            12, 0.99, 0.4, 0.15, 65, 1.4, 0.29, 1.48,
        ]
        metrics = self._read(core + [9.9] * 10)
        self.assertEqual(metrics["avg_top1_prob"], 0.6)
        self.assertTrue(metrics["teacher_forward_kl_t1"] != metrics["teacher_forward_kl_t1"])


if __name__ == "__main__":
    unittest.main()
