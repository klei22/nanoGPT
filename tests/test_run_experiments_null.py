import unittest
from pathlib import Path

from optimization_and_search.run_experiments import (
    build_command,
    format_run_name,
    generate_combinations,
    load_configurations,
)


class NullExperimentValueTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config_path = Path(__file__).parents[1] / 'explorations' / 'test_null.yaml'
        cls.configs = load_configurations(config_path, 'yaml')

    def test_yaml_null_keeps_default(self):
        combos = [
            combo
            for config in self.configs
            for combo, _ in generate_combinations(config)
        ]
        default_combo = next(combo for combo in combos if combo['norm_variant_wte'] is None)
        override_combo = next(combo for combo in combos if combo['norm_variant_wte'] == 'rmsnorm')

        self.assertNotIn('--norm_variant_wte', build_command(default_combo))
        self.assertIn('--norm_variant_wte', build_command(override_combo))

    def test_run_name_labels_null_as_default(self):
        self.assertEqual(
            format_run_name({'norm_variant_wte': None}, 'base', ''),
            'base-default',
        )


if __name__ == '__main__':
    unittest.main()
