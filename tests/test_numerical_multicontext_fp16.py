import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt_conf import GPTConfig
from model import GPT
from variations.numerical_mapping_variations import get_numerical_embedding


class CaptureEmbedding(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.n_embd = n_embd
        self.last_input = None
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        self.last_input = x.detach().clone()
        return x.repeat(1, 1, self.n_embd)


class ZeroOutput(nn.Module):
    def forward(self, x):
        return torch.zeros((*x.shape[:2], 1), dtype=x.dtype, device=x.device)


class NumericalMulticontextFP16Test(unittest.TestCase):
    def test_fp16bits_to_fp32_matches_numpy_reference(self):
        bit_patterns = torch.tensor(
            [0x0000, 0x8000, 0x0001, 0x3C00, 0xBC00, 0x7C00, 0xFC00, 0x7E00],
            dtype=torch.int64,
        )

        decoded = GPT._fp16bits_to_fp32(bit_patterns)
        expected = np.array(bit_patterns.tolist(), dtype=np.uint16).view(np.float16).astype(np.float32)
        expected_t = torch.from_numpy(expected)

        finite_mask = torch.isfinite(expected_t)
        self.assertTrue(torch.allclose(decoded[finite_mask], expected_t[finite_mask], atol=1e-7, rtol=0.0))
        self.assertTrue(torch.equal(torch.isinf(decoded), torch.isinf(expected_t)))
        self.assertTrue(torch.equal(torch.isnan(decoded), torch.isnan(expected_t)))

    def test_numerical_embedding_channel_norm_hyperspherenorm(self):
        cfg = GPTConfig(
            block_size=4,
            vocab_size=16,
            n_layer=0,
            n_head=1,
            n_embd=6,
            numerical_embedding_variant="linear",
            norm_channel_variant="hyperspherenorm",
            norm_channel_radius=2.5,
            norm_channel_scale=1.0,
            norm_channel_gain=False,
            norm_channel_radius_learning=False,
        )
        embedding = get_numerical_embedding(cfg)

        x = torch.randn(3, 4, 1)
        out = embedding(x)
        norms = out.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.full_like(norms, 2.5), atol=1e-4, rtol=1e-4))

    def test_numerical_multicontext_fp16_decodes_inputs_and_targets(self):
        cfg = GPTConfig(
            block_size=4,
            vocab_size=65536,
            n_layer=0,
            n_head=1,
            n_embd=8,
            dropout=0.0,
            use_abs_pos_embeddings=False,
            multicontext=True,
            numerical_multicontext=True,
            numerical_multicontext_input_format="fp16_bits",
            vocab_sizes=[65536],
        )
        model = GPT(cfg)

        capture_embedding = CaptureEmbedding(n_embd=cfg.n_embd)
        model.numerical_embeddings["0"] = capture_embedding
        model.numerical_output_mlps["0"] = ZeroOutput()

        input_fp16 = np.array([[0.5, -1.5, 2.0, -0.25]], dtype=np.float16)
        target_fp16 = np.array([[0.25, -0.75, 0.5, -1.0]], dtype=np.float16)

        token_bits = torch.from_numpy(input_fp16.view(np.uint16).astype(np.int64))
        target_bits = torch.from_numpy(target_fp16.view(np.uint16).astype(np.int64))

        _, losses = model(
            idx=None,
            token_dict={"ctx": token_bits},
            target_dict={"ctx": target_bits},
        )

        self.assertIsNotNone(losses)
        decoded_inputs = GPT._fp16bits_to_fp32(token_bits)
        self.assertIsNotNone(capture_embedding.last_input)
        assert capture_embedding.last_input is not None
        self.assertTrue(
            torch.allclose(
                capture_embedding.last_input.squeeze(-1),
                decoded_inputs,
                atol=1e-6,
                rtol=0.0,
            )
        )

        decoded_targets = GPT._fp16bits_to_fp32(target_bits)
        expected_loss = F.huber_loss(
            torch.zeros_like(decoded_targets),
            decoded_targets,
            delta=1.0,
            reduction="mean",
        )
        self.assertTrue(torch.allclose(losses[0], expected_loss.to(losses[0].dtype), atol=1e-6, rtol=0.0))


if __name__ == "__main__":
    unittest.main()
