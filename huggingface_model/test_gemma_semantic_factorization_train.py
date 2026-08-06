from huggingface_model.gemma_semantic_factorization_train import (
    ALL_CAPS_FEATURE,
    CAPITALIZED_FEATURE,
    SPACE_FEATURE,
    build_factorized_vocabulary,
    canonicalize_token,
)


def test_canonicalize_token_composes_space_and_case_features():
    vocabulary = {"hello", "Hello", "▁Hello", "USA", "usa"}

    assert canonicalize_token("▁Hello", vocabulary, "▁") == (
        "hello",
        SPACE_FEATURE | CAPITALIZED_FEATURE,
    )
    assert canonicalize_token("USA", vocabulary, "▁") == ("usa", ALL_CAPS_FEATURE)


def test_build_factorized_vocabulary_preserves_reserved_head_rows():
    vocabulary = {"hello": 0, "Hello": 1, "▁Hello": 2, "USA": 4, "usa": 5}

    factorized = build_factorized_vocabulary(vocabulary, head_size=6)

    assert factorized.original_size == 6
    assert factorized.base_size == 3
    assert factorized.removed == 3
    assert factorized.id_map[0].base_id == factorized.id_map[1].base_id == factorized.id_map[2].base_id
    assert factorized.id_map[2].feature_mask == SPACE_FEATURE | CAPITALIZED_FEATURE
    assert factorized.id_map[3].feature_mask == 0
    assert factorized.base_tokens[factorized.id_map[3].base_id] == "<reserved:3>"
