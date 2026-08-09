from pathlib import Path

import pytest
import yaml

from hyperparam_search import dict_to_cli, parse_override_value, patched_argv


def test_dict_to_cli_forwards_false_booleans_as_no_flags():
    cli = dict_to_cli(
        {
            "use_rotary_embeddings": True,
            "use_abs_pos_embeddings": False,
            "n_embd": 64,
            "mlp_size_layerlist": [128, 256],
            "_private": "skip-me",
        }
    )

    assert "--use_rotary_embeddings" in cli
    assert "--no-use_abs_pos_embeddings" in cli
    assert "--use_abs_pos_embeddings" not in cli
    assert "--_private" not in cli
    list_start = cli.index("--mlp_size_layerlist") + 1
    assert cli[list_start : list_start + 2] == ["128", "256"]


def test_demo_yaml_abs_pos_false_survives_train_argparse():
    pytest.importorskip("torch")
    from train_args import parse_args

    cfg = yaml.safe_load(Path("hp_searches/efficiency_targets_demo.yaml").read_text())
    with patched_argv(["train.py", *dict_to_cli(cfg)]):
        args, *_ = parse_args()

    assert args.use_rotary_embeddings is True
    assert args.use_abs_pos_embeddings is False


def test_override_values_accept_yaml_style_false():
    assert parse_override_value("false") is False
    assert parse_override_value("true") is True
    assert parse_override_value("[1, 2, 3]") == [1, 2, 3]
