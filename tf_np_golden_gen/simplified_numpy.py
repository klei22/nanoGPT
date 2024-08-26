import numpy as np
import pickle
import math

class GPTConfig:
    def __init__(self, vocab_size=65, block_size=256, n_layer=6, n_head=6, n_embd=384, n_kv_group=6):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.n_kv_group = n_kv_group

class RMSNorm:
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.dim = dim
        self.weight = np.ones((dim,))

    def __call__(self, x):
        mean_square = np.mean(x ** 2, axis=-1, keepdims=True)
        normed = x / np.sqrt(mean_square + self.eps)
        return normed * self.weight

class Linear:
    def __init__(self, in_features, out_features, weights):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = weights  # Shape: (out_features, in_features)

    def __call__(self, x):
        return np.dot(x, self.weight.T)

class CausalSelfAttention:
    def __init__(self, config, weights):
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.scale = 1.0 / math.sqrt(config.n_embd // config.n_head)
        self.n_kv_group = config.n_kv_group

        # Load weights for q, k, v projections and output projection
        self.c_attn_q = Linear(config.n_embd, config.n_embd, weights['attn.c_attn_q.weight'])
        self.c_attn_k = Linear(config.n_embd, config.n_kv_group * (config.n_embd // config.n_head), weights['attn.c_attn_k.weight'])
        self.c_attn_v = Linear(config.n_embd, config.n_kv_group * (config.n_embd // config.n_head), weights['attn.c_attn_v.weight'])
        self.c_proj = Linear(config.n_embd, config.n_embd, weights['attn.c_proj.weight'])

    def __call__(self, x):
        B, T, C = x.shape

        q = self.c_attn_q(x).reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)
        k = self.c_attn_k(x).reshape(B, T, self.n_kv_group, C // self.n_head).transpose(0, 2, 1, 3)
        v = self.c_attn_v(x).reshape(B, T, self.n_kv_group, C // self.n_head).transpose(0, 2, 1, 3)

        att = np.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        att = np.tril(np.ones((T, T)))  # Causal mask
        att = att / np.sum(att, axis=-1, keepdims=True)

        y = np.matmul(att, v).transpose(0, 2, 1, 3).reshape(B, T, C)
        y = self.c_proj(y)
        return y

class MLP:
    def __init__(self, config, weights):
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, weights['mlp.c_fc.weight'])
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, weights['mlp.c_proj.weight'])

    def __call__(self, x):
        x = np.maximum(0, self.c_fc(x))  # ReLU activation
        x = self.c_proj(x)
        return x

class Block:
    def __init__(self, config, weights):
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, weights)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config, weights)

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT:
    def __init__(self, config, weights):
        self.config = config
        
        # Initialize weights correctly based on loaded keys
        self.wte = weights['transformer.wte.weight']  # Embedding weights
        self.blocks = [Block(config, {k.split(f'transformer.h.{i}.')[-1]: weights[k] for k in weights if f'transformer.h.{i}.' in k}) for i in range(config.n_layer)]
        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = weights['lm_head.weight']

    def __call__(self, idx, max_new_tokens):
        B, T = idx.shape
        x = self.wte[idx]  # Token embedding lookup
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = np.dot(x, self.lm_head.T)  # Final linear projection

        for _ in range(max_new_tokens):
            idx_next = np.argmax(logits[:, -1, :], axis=-1)
            idx = np.concatenate([idx, idx_next[:, None]], axis=1)

        return idx

def load_weights(weights_path):
    with open(weights_path, 'rb') as f:
        weights = pickle.load(f)
    return weights

def main():
    config = GPTConfig()
    weights = load_weights('weights.pkl')
    model = GPT(config, weights)

    # Example inference
    start_token = 39  # 'a' (example)
    idx = np.array([[start_token]])
    result = model(idx, max_new_tokens=20)
    print("Generated token indices:", result)

if __name__ == "__main__":
    main()

