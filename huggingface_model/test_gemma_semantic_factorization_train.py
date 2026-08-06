from huggingface_model.gemma_semantic_factorization_train import (
    ALL_CAPS_FEATURE,
    CAPITALIZED_FEATURE,
    SPACE_FEATURE,
    add_feature_embeddings,
    apply_feature_head,
    build_factorized_vocabulary,
    canonicalize_token,
    token_loss_records,
    update_token_loss_totals,
)


def test_auxiliary_features_follow_consumer_dtype():
    class FakeTensor:
        def __init__(self, dtype):
            self.dtype = dtype

        def to(self, *, dtype):
            return FakeTensor(dtype)

        def __add__(self, other):
            assert other.dtype == self.dtype
            return FakeTensor(self.dtype)

    class FakeHead:
        weight = FakeTensor("float32")

        def __call__(self, hidden_states):
            assert hidden_states.dtype == self.weight.dtype
            return hidden_states

    token_embeddings = FakeTensor("bfloat16")
    feature_embeddings = FakeTensor("float32")
    combined = add_feature_embeddings(token_embeddings, feature_embeddings)
    logits = apply_feature_head(FakeHead(), combined)

    assert combined.dtype == "bfloat16"
    assert logits.dtype == "float32"


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


def test_token_loss_records_track_observed_original_and_factorized_tokens():
    factorized = build_factorized_vocabulary({"hello": 0, "Hello": 1}, head_size=2)
    totals = {}

    update_token_loss_totals(totals, [1, 1, 0], [3.0, 1.0, 0.5])
    records = token_loss_records(totals, {0: "hello", 1: "Hello"}, factorized)

    assert records == [
        {
            "token_id": 0,
            "token": "hello",
            "mean_loss": 0.5,
            "count": 1,
            "base_id": 0,
            "base_token": "hello",
            "feature_mask": 0,
        },
        {
            "token_id": 1,
            "token": "Hello",
            "mean_loss": 2.0,
            "count": 2,
            "base_id": 0,
            "base_token": "hello",
            "feature_mask": CAPITALIZED_FEATURE,
        },
    ]
