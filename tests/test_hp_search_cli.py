import argparse
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if "torch" not in sys.modules:
    torch_stub = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            empty_cache=lambda: None,
        )
    )
    sys.modules["torch"] = torch_stub

from hyperparam_search import dict_to_cli as controller_dict_to_cli
from optimization_and_search.run_hp_trials import dict_to_cli as worker_dict_to_cli


def test_dict_to_cli_emits_boolean_optional_false_flags():
    config = {
        "use_abs_pos_embeddings": False,
        "use_rotary_embeddings": True,
        "_private_search_state": False,
        "n_layer": 1,
    }

    for dict_to_cli in (controller_dict_to_cli, worker_dict_to_cli):
        cli = dict_to_cli(config)
        assert "--no-use_abs_pos_embeddings" in cli
        assert "--use_rotary_embeddings" in cli
        assert "--_private_search_state" not in cli
        assert "--no-_private_search_state" not in cli
        assert "--n_layer" in cli


def test_false_boolean_cli_overrides_boolean_optional_defaults():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_abs_pos_embeddings",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--use_rotary_embeddings",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    config = {"use_abs_pos_embeddings": False, "use_rotary_embeddings": True}

    parsed = parser.parse_args(controller_dict_to_cli(config))

    assert parsed.use_abs_pos_embeddings is False
    assert parsed.use_rotary_embeddings is True
