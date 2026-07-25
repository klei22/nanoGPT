import unittest
from pathlib import Path

from optimization_and_search.run_experiments import (
    DEFAULT_VALUE,
    build_command,
    format_run_name,
    generate_combinations,
    load_configurations,
)


class NullNoneExperimentValuesTest(unittest.TestCase):
    def test_yaml_null_keeps_default_and_none_sets_none(self):
        config_path = Path(__file__).parents[1] / 'explorations' / 'test_null_none.yaml'
        configs = load_configurations(config_path, 'yaml')

        combos = [combo for combo, _ in generate_combinations(configs[0])]
        values = [combo['property'] for combo in combos]
        self.assertEqual(values, [None, DEFAULT_VALUE, '1', '2'])

        commands = [build_command(combo) for combo in combos]
        self.assertEqual(commands[0], ['python3', 'train.py', '--property', 'none'])
        self.assertEqual(commands[1], ['python3', 'train.py'])
        self.assertEqual(commands[2], ['python3', 'train.py', '--property', '1'])
        self.assertEqual(commands[3], ['python3', 'train.py', '--property', '2'])

    def test_run_names_distinguish_default_and_none(self):
        self.assertEqual(format_run_name({'property': DEFAULT_VALUE}, 'base', ''), 'base-default')
        self.assertEqual(format_run_name({'property': None}, 'base', ''), 'base-none')


if __name__ == '__main__':
    unittest.main()
