from types import SimpleNamespace

import pytest
import torch

from app.model_service import TokenInfo, layernorm_analysis, regex_search_tokens


def _assets(*, unit_offset=False, kind="rmsnorm"):
    tokens = [
        TokenInfo(0, "▁cat", " cat", "cat"),
        TokenInfo(1, "▁dog", " dog", "dog"),
    ]
    weight = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 0.0, 2.0]])
    return SimpleNamespace(
        weight=weight,
        hidden_dim=3,
        layernorms={"model.norm.weight": torch.tensor([0.5, 2.0, 1.0])},
        norm_epsilon=0.0,
        norm_kind=kind,
        norm_unit_offset=unit_offset,
        token_infos=tokens,
        token=lambda token_id: tokens[token_id],
    )


def test_layernorm_analysis_uses_one_gain_sorted_channel_permutation():
    result = layernorm_analysis(_assets(), "model.norm.weight", [0, 1])

    assert result["channel_indices"] == [1, 2, 0]
    assert result["gains"] == [2.0, 1.0, 0.5]
    assert result["embeddings"][0]["before"] == [2.0, 3.0, 1.0]
    rms = ((1.0 + 4.0 + 9.0) / 3.0) ** 0.5
    assert result["embeddings"][0]["after"] == pytest.approx([4.0 / rms, 3.0 / rms, 0.5 / rms])
    assert result["embeddings"][0]["before_magnitude"] == pytest.approx(14.0**0.5)
    assert result["embeddings"][0]["before_participation_ratio"] == pytest.approx(14.0**2 / 98.0)
    assert result["embeddings"][0]["after_participation_ratio"] > 0
    assert result["embeddings"][0]["norm_rotation_deg"] >= 0
    assert result["relative_angle_delta_deg"] == pytest.approx(
        result["after_pair_angle_deg"] - result["before_pair_angle_deg"]
    )


def test_gemma_unit_offset_is_reported_and_applied():
    result = layernorm_analysis(_assets(unit_offset=True), "model.norm.weight", [0, 1])
    assert result["unit_offset"] is True
    assert result["gains"] == [3.0, 2.0, 1.5]


def test_regex_search_matches_raw_or_display_and_rejects_bad_patterns():
    assets = _assets()
    assert [token.token_id for token in regex_search_tokens(assets, "^(▁cat| dog)$")] == [0, 1]
    with pytest.raises(ValueError, match="Invalid regular expression"):
        regex_search_tokens(assets, "[")


def test_attention_head_operator_returns_matrix_and_vector_diagnostics():
    assets = _assets()
    assets.attention_projections = {
        "model.layers.0.self_attn.q_proj.weight": torch.eye(3),
        "model.layers.0.self_attn.k_proj.weight": torch.eye(3),
    }
    assets.num_attention_heads = 1
    assets.num_key_value_heads = 1
    assets.head_dim = 3

    result = layernorm_analysis(
        assets, "model.norm.weight", [0, 1], "model.layers.0.self_attn", 0
    )

    assert result["attention"]["head"] == 0
    assert len(result["attention"]["embeddings"]) == 2
    assert result["attention"]["matrix_stats"]["skew_fraction"] == pytest.approx(0.0)
    assert result["attention"]["matrix_stats"]["rotation_residual"] >= 0
    assert result["attention"]["embeddings"][0]["magnitude"] > 0
    expected_after = torch.tensor(result["embeddings"][0]["after"])
    assert result["attention"]["embeddings"][0]["dot_product_with_norm"] == pytest.approx(
        torch.dot(expected_after, expected_after).item()
    )
    assert result["attention"]["pipeline"] == "input_embedding -> selected_norm_with_gain -> WqWkT"
