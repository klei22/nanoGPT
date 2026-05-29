"""Tests for Token Superposition Training (TST).

Covers the two algorithmic pieces of TST:
  * the multi-hot cross-entropy (MCE) "next bag-of-tokens" loss, and
  * the bag-of-token-embeddings input superposition in ``GPT.forward``.
"""

import math
import unittest

import torch
import torch.nn.functional as F

from gpt_conf import GPTConfig
from model import GPT
from train_variations.loss_variants import (
    LOSS_VARIANTS,
    build_superposition_loss_function,
    multi_hot_cross_entropy,
    superposition_bag_weights,
)


def _reference_mce(logits, targets, weighting="uniform", ignore_index=-1):
    """Independent re-implementation of the paper's Listing 3 for cross-checking."""
    bs, seq, vocab = logits.shape
    bag = targets.shape[-1] // seq
    offset = bag - 1
    flat = logits.reshape(bs * seq, vocab).float()
    padded = F.pad(targets, (0, offset), mode="constant", value=ignore_index)[..., offset:]
    padded = padded.reshape(bs, seq, bag)
    weights = superposition_bag_weights(bag, weighting)
    loss = 0.0
    for i, w in enumerate(weights):
        if w == 0.0:
            continue
        loss = loss + w * F.cross_entropy(flat, padded[..., i].reshape(-1), ignore_index=ignore_index)
    return loss / sum(weights)


class MultiHotCrossEntropyTest(unittest.TestCase):
    def test_registered_as_loss_variant(self):
        self.assertIn("multi_hot_cross_entropy", LOSS_VARIANTS)

    def test_bag_weights_match_paper(self):
        self.assertEqual(superposition_bag_weights(3, "uniform"), [1.0, 1.0, 1.0])
        self.assertEqual(superposition_bag_weights(3, "power_law"), [1.0, 0.5, 1.0 / 3.0])
        self.assertEqual(
            superposition_bag_weights(2, "exponential"), [math.exp(-1), math.exp(-2)]
        )
        self.assertEqual(superposition_bag_weights(3, "first_token"), [1.0, 0.0, 0.0])

    def test_reduces_to_cross_entropy_when_bag_size_one(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 5, 7)
        targets = torch.randint(0, 7, (2, 5))
        mce = multi_hot_cross_entropy(logits, targets)
        ce = F.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1), ignore_index=-1)
        self.assertTrue(torch.allclose(mce, ce, atol=1e-6))

    def test_infers_bag_size_and_matches_reference(self):
        torch.manual_seed(1)
        bs, seq, vocab, bag = 3, 4, 11, 4
        logits = torch.randn(bs, seq, vocab)
        targets = torch.randint(0, vocab, (bs, seq * bag))
        for weighting in ("uniform", "power_law", "exponential", "first_token"):
            got = multi_hot_cross_entropy(logits, targets, weighting=weighting)
            expected = _reference_mce(logits, targets, weighting=weighting)
            self.assertTrue(
                torch.allclose(got, expected, atol=1e-6),
                msg=f"weighting={weighting}",
            )

    def test_first_token_equals_next_token_after_bag(self):
        """With first_token weighting, MCE == CE against the immediate next token."""
        torch.manual_seed(2)
        bs, seq, vocab, bag = 2, 6, 9, 3
        logits = torch.randn(bs, seq, vocab)
        targets = torch.randint(0, vocab, (bs, seq * bag))
        got = multi_hot_cross_entropy(logits, targets, weighting="first_token")
        # The token right after each bag lives at offset s-1, every s tokens.
        next_token_labels = targets[:, bag - 1 :: bag]
        expected = F.cross_entropy(
            logits.reshape(-1, vocab), next_token_labels.reshape(-1), ignore_index=-1
        )
        self.assertTrue(torch.allclose(got, expected, atol=1e-6))

    def test_uniform_on_uniform_logits_is_log_vocab(self):
        bs, seq, vocab, bag = 2, 5, 8, 4
        logits = torch.zeros(bs, seq, vocab)
        targets = torch.randint(0, vocab, (bs, seq * bag))
        loss = multi_hot_cross_entropy(logits, targets, weighting="uniform")
        self.assertTrue(torch.allclose(loss, torch.tensor(math.log(vocab)), atol=1e-5))

    def test_causal_padding_ignores_overflow(self):
        """The tail of the last bag has no real targets and must be ignored."""
        vocab, bag = 5, 3
        logits = torch.randn(1, 2, vocab)
        targets = torch.randint(0, vocab, (1, 2 * bag))
        # Manually: position 1's bag = [t[3], t[4], t[5]]; only t[3] (=t[1*bag+...])
        # Build the expected with explicit ignore handling via the reference.
        got = multi_hot_cross_entropy(logits, targets, weighting="uniform")
        expected = _reference_mce(logits, targets, weighting="uniform")
        self.assertTrue(torch.allclose(got, expected, atol=1e-6))

    def test_builder_uses_configured_weighting(self):
        class _Args:
            superposition_weighting = "power_law"

        loss_fn = build_superposition_loss_function(_Args())
        torch.manual_seed(3)
        logits = torch.randn(1, 3, 6)
        targets = torch.randint(0, 6, (1, 6))
        got = loss_fn(logits, targets, iter_num=0)
        expected = multi_hot_cross_entropy(logits, targets, weighting="power_law")
        self.assertTrue(torch.allclose(got, expected, atol=1e-6))


def _tiny_model(block_size=8, vocab_size=16, n_embd=8):
    cfg = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=2,
        n_head=2,
        n_embd=n_embd,
        dropout=0.0,
        bias=False,
        use_abs_pos_embeddings=False,
    )
    model = GPT(cfg)
    model.eval()  # disable dropout / embedding noise for deterministic checks
    return model


class InputSuperpositionForwardTest(unittest.TestCase):
    def test_three_dim_input_produces_latent_length_logits(self):
        torch.manual_seed(0)
        model = _tiny_model(block_size=8, vocab_size=16)
        bag = 4
        latent_len = 8
        idx = torch.randint(0, 16, (3, latent_len, bag))
        with torch.no_grad():
            logits, loss = model(idx)
        # Inference path returns only the last position's logits.
        self.assertEqual(logits.shape, (3, 1, 16))
        self.assertIsNone(loss)

    def test_identical_bags_match_plain_token_forward(self):
        """Averaging s identical embeddings equals embedding that single token."""
        torch.manual_seed(0)
        model = _tiny_model(block_size=8, vocab_size=16)
        bag = 3
        idx2d = torch.randint(0, 16, (2, 8))
        targets = torch.randint(0, 16, (2, 8))
        idx3d = idx2d.unsqueeze(-1).repeat(1, 1, bag)
        with torch.no_grad():
            logits2d, _ = model(idx2d, targets=targets)
            logits3d, _ = model(idx3d, targets=targets)
        self.assertEqual(logits3d.shape, logits2d.shape)
        self.assertTrue(torch.allclose(logits3d, logits2d, atol=1e-5))

    def test_superposition_training_step_shapes_and_grad(self):
        """End-to-end: fold tokens into bags (as get_batch does), forward, MCE, backward."""
        torch.manual_seed(0)
        model = _tiny_model(block_size=6, vocab_size=16)
        model.train()
        bag = 4
        block_size = 6
        batch = 2
        span = block_size * bag
        # Emulate get_batch: contiguous span of data folded into bags.
        x = torch.randint(0, 16, (batch, span)).view(batch, block_size, bag)
        y = torch.randint(0, 16, (batch, span))
        loss_fn = build_superposition_loss_function(
            type("A", (), {"superposition_weighting": "uniform"})()
        )
        logits, loss = model(x, targets=y, loss_fn=loss_fn)
        self.assertEqual(logits.shape, (batch, block_size, 16))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(len(grads) > 0)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))


if __name__ == "__main__":
    unittest.main()
