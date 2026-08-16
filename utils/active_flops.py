"""Transparent architecture-aware FLOP estimates (multiply-add counts as two)."""


def estimate_transformer_flops(*, layers, sequence_length, streams=1, width=384,
                               mlp_expansion=4, vocab_size=50304, active_heads=1):
    projection = layers * streams * sequence_length * 2 * (
        4 * width * width + 2 * mlp_expansion * width * width
    )
    attention = layers * streams * 4 * sequence_length * sequence_length * width
    vocabulary = 2 * sequence_length * streams * active_heads * width * vocab_size
    return {"transformer_projection_mlp": projection, "attention_score_value": attention,
            "vocabulary_head": vocabulary, "total": projection + attention + vocabulary}


def parameter_counts(module, active_parameter_names=None):
    total = sum(parameter.numel() for parameter in module.parameters())
    if active_parameter_names is None:
        active = total
    else:
        names = set(active_parameter_names)
        active = sum(parameter.numel() for name, parameter in module.named_parameters() if name in names)
    return {"total_parameters": total, "active_parameters": active}
