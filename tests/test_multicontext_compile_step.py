import unittest

import torch

from gpt_conf import GPTConfig
from model import GPT


class MulticontextCompileStepTest(unittest.TestCase):
    def _build_model(self):
        cfg = GPTConfig(
            block_size=8,
            vocab_size=16,
            n_layer=1,
            n_head=2,
            n_kv_group=2,
            n_embd=8,
            dropout=0.0,
            multicontext=True,
            vocab_sizes=[11, 13],
            use_abs_pos_embeddings=False,
            disable_flash_attention=True,
        )
        model = GPT(cfg)
        model.eval()
        return model

    def test_generate_multicontext_step_matches_forward_logits_list(self):
        torch.manual_seed(1234)
        model = self._build_model()
        token_dict = {
            "ctx_a": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "ctx_b": torch.tensor([[4, 5, 6]], dtype=torch.long),
        }

        forward_logits, forward_losses = model(None, token_dict=token_dict, target_dict=None)
        step_logits, step_losses = model.generate_multicontext_step(token_dict)

        self.assertIsNone(forward_losses)
        self.assertIsNone(step_losses)
        self.assertEqual(len(step_logits), len(forward_logits))
        self.assertEqual([tuple(logits.shape) for logits in step_logits], [(1, 1, 11), (1, 1, 13)])
        for expected, actual in zip(forward_logits, step_logits):
            self.assertTrue(torch.allclose(expected, actual, atol=0.0, rtol=0.0))

    def test_compiled_generate_multicontext_step_preserves_logits_list_shapes(self):
        torch.manual_seed(5678)
        model = self._build_model()
        token_dict = {
            "ctx_a": torch.tensor([[5]], dtype=torch.long),
            "ctx_b": torch.tensor([[9]], dtype=torch.long),
        }

        eager_logits, eager_losses = model.generate_multicontext_step(token_dict)
        compiled_step = torch.compile(model.generate_multicontext_step, backend="eager")
        compiled_logits, compiled_losses = compiled_step(token_dict)

        self.assertIsNone(eager_losses)
        self.assertIsNone(compiled_losses)
        self.assertEqual(len(compiled_logits), len(eager_logits))
        self.assertEqual(
            [tuple(logits.shape) for logits in compiled_logits],
            [tuple(logits.shape) for logits in eager_logits],
        )
        for expected, actual in zip(eager_logits, compiled_logits):
            self.assertTrue(torch.allclose(expected, actual, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
