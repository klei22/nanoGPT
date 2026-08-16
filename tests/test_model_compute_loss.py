import torch

from gpt_conf import GPTConfig
from model import GPT


def test_compute_loss_false_returns_full_logits_without_cross_entropy():
    config = GPTConfig(
        block_size=8,
        vocab_size=17,
        n_layer=1,
        n_head=1,
        n_embd=8,
        dropout=0.0,
    )
    model = GPT(config)
    tokens = torch.randint(0, config.vocab_size, (2, 4))
    targets = torch.randint(0, config.vocab_size, (2, 4))

    logits, loss = model(tokens, targets=targets, compute_loss=False)

    assert logits.shape == (2, 4, config.vocab_size)
    assert loss is None
