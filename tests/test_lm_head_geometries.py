import unittest

import torch

from gpt_conf import GPTConfig
from model import GPT


class LMHeadGeometryTest(unittest.TestCase):
    def _run_geometry(self, geometry: str):
        torch.manual_seed(0)
        config = GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            use_abs_pos_embeddings=False,
            lm_head_geometry=geometry,
            lm_head_logit_scale=1.0,
            lm_head_hyperbolic_c=1.0,
        )
        model = GPT(config)
        idx = torch.randint(0, config.vocab_size, (2, 5))
        targets = torch.randint(0, config.vocab_size, (2, 5))
        logits, loss = model(idx, targets=targets)

        self.assertEqual(tuple(logits.shape), (2, 5, config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(logits).all().item())
        self.assertTrue(torch.isfinite(loss).item())

    def test_euclidean_geometry(self):
        self._run_geometry("euclidean")

    def test_hyperspherical_geometry(self):
        self._run_geometry("hyperspherical")

    def test_hyperbolic_geometry(self):
        self._run_geometry("hyperbolic")


if __name__ == "__main__":
    unittest.main()
