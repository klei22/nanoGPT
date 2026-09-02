from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]


def test_smoke_manifest_has_paired_non_cartesian_arms():
    config = yaml.safe_load((ROOT / "explorations/predictive_width_smoke.yaml").read_text())
    groups = {group["named_group"]: group for group in config["named_static_groups"]}
    assert set(groups) == {"dense", "s4_direct", "s4_collapsed"}
    assert groups["s4_direct"]["predictive_stream_dim"] == [32]
    assert groups["s4_collapsed"]["predictive_stream_dim"] == [32]
    assert groups["s4_direct"]["predictive_stream_n_head"] == [2]
    assert groups["s4_collapsed"]["predictive_stream_n_head"] == [2]

    arms = config["parameter_groups"]
    assert [arm["named_group_static"] for arm in arms] == [
        ["dense"], ["s4_direct"], ["s4_collapsed"]
    ]
    assert all(arm["seed"] == [1337] for arm in arms)
    assert config["common_group"]["data_seed"] == [91001]
    assert config["common_group"]["eval_seed"] == [92001]


def test_demo_is_executable_and_references_manifest():
    script = ROOT / "demos/predictive_width_experiments.sh"
    assert script.stat().st_mode & 0o111
    assert "explorations/predictive_width_smoke.yaml" in script.read_text()
