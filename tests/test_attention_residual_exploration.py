from pathlib import Path

import yaml


EXPLORATION_PATH = (
    Path(__file__).parents[1]
    / "explorations"
    / "attention_residual_weighting_comparison.yaml"
)


def test_attention_residual_comparison_avoids_optional_tensorboard_dependency():
    config = yaml.safe_load(EXPLORATION_PATH.read_text())

    assert config["common_group"]["tensorboard_log"] == [False]


def test_relu2max_comparison_uses_unit_divisor_and_initialization_sweep():
    config = yaml.safe_load(EXPLORATION_PATH.read_text())
    relu2max_group = config["parameter_groups"][1]

    assert relu2max_group["attention_residual_weighting"] == ["relu2max"]
    assert relu2max_group["relu2max_divisor"] == [1.0]
    assert relu2max_group["attention_residual_use_qk_norm"] == [False, True]
    assert relu2max_group["attention_residual_use_qk_norm_scale"] == {
        "conditions": {"attention_residual_use_qk_norm": True},
        "options": [False, True],
    }
    assert relu2max_group["attention_residual_relu2max_query_init_scale"] == [
        1.0,
        0.1,
        0.02,
    ]
