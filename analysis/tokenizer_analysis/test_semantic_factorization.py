import json

import pytest

from semantic_factorization import analyze_tokens, load_tokenizer_vocabulary


def test_factorization_rules_and_overlap():
    tokens = {"hello", "▁hello", "Hello", "▁Hello", "NASA", "nasa", "▁NASA", "unrelated"}
    result = analyze_tokens(tokens)
    assert result.leading_space == {"▁hello", "▁Hello", "▁NASA"}
    assert result.initial_capital == {"Hello", "▁Hello"}
    # ▁NASA has no ▁nasa counterpart, so it is removable only by SPACE here.
    assert result.all_caps == {"NASA"}
    assert result.exclusive_counts() == (3, 1, 1)
    assert len(result.removed) == 5
    assert result.factorized_vocabulary_size == 3


def test_all_caps_requires_a_cased_character_and_counterpart():
    assert not analyze_tokens({"123", "!", "ABC", "mixed", "▁MISSING"}).all_caps


def test_load_tokenizer_vocabulary(tmp_path):
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps({"model": {"vocab": {"a": 0, "A": 1}}}), encoding="utf-8")
    assert load_tokenizer_vocabulary(path) == {"a": 0, "A": 1}


def test_load_tokenizer_vocabulary_rejects_wrong_shape(tmp_path):
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps({"model": {"vocab": []}}), encoding="utf-8")
    with pytest.raises(ValueError, match="model.vocab"):
        load_tokenizer_vocabulary(path)
