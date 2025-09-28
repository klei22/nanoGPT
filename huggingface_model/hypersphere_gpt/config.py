from transformers import PretrainedConfig


class HypersphereGPTConfig(PretrainedConfig):
    model_type = "hypersphere-gpt"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 50257,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        n_inner: int | None = None,
        block_size: int = 2048,
        dropout: float = 0.0,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        use_qk_norm: bool = True,
        use_qk_norm_scale: bool = True,
        use_rotary_embeddings: bool = True,
        rope_length: int | None = None,
        hsnorm_gain: bool = True,
        hsnorm_radius: float | None = None,
        hsnorm_radius_learning: bool = False,
        use_peri_ln_attn: bool = True,
        use_peri_ln_mlp: bool = True,
        bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            pad_token_id=vocab_size - 1,
            bos_token_id=50256,
            eos_token_id=50256,
            tie_word_embeddings=True,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.n_inner = n_inner if n_inner is not None else 4 * n_embd
        self.block_size = block_size
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_qk_norm = use_qk_norm
        self.use_qk_norm_scale = use_qk_norm_scale
        self.use_rotary_embeddings = use_rotary_embeddings
        self.rope_length = rope_length
        self.hsnorm_gain = hsnorm_gain
        self.hsnorm_radius = hsnorm_radius
        self.hsnorm_radius_learning = hsnorm_radius_learning
        self.use_peri_ln_attn = use_peri_ln_attn
        self.use_peri_ln_mlp = use_peri_ln_mlp
        self.bias = bias
        self.use_cache = kwargs.get("use_cache", True)
