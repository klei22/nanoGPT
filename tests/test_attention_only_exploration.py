from pathlib import Path

import yaml


def test_opus_attention_only_exploration_has_matched_comparison_arms():
    config_path = (
        Path(__file__).parents[1] / "explorations" / "attention_only_san.yaml"
    )
    config = yaml.safe_load(config_path.read_text())

    assert config["common_group"]["dataset"] == ["opus-100"]
    assert config["common_group"]["seed"] == [42, 43, 44]

    san, transformer = config["parameter_groups"]
    assert san["attention_only"] == [True]
    assert san["n_layer"] == [20]

    assert transformer["attention_only"] == [False]
    assert transformer["n_layer"] == [4]
    assert transformer["mlp_variant"] == ["swiglu"]
    assert transformer["mlp_size"] == [2048]

    # Each arm must tune its own learning rate rather than inheriting one rate.
    assert len(san["learning_rate"]) > 1
    assert len(transformer["learning_rate"]) > 1
