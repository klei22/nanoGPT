import unittest

import torch

from gpt_conf import GPTConfig
from variations.mlp_variations import get_mlp_instance


class QuadPathMagnitudeSwigluTest(unittest.TestCase):
    def test_forward_shape_matches_embedding_dim(self):
        config = GPTConfig(
            n_embd=16,
            n_layer=1,
            n_head=1,
            mlp_variant="quad_path_magnitude_swiglu",
            mlp_expansion_factor=2,
            dropout=0.0,
            bias=False,
        )
        mlp = get_mlp_instance(config)

        x = torch.randn(3, 5, config.n_embd)
        out = mlp(x)

        self.assertEqual(out.shape, x.shape)
        self.assertEqual(mlp.c_proj_pos_neg.in_features, mlp.c_proj_neg_neg.in_features)
        self.assertEqual(mlp.c_proj_pos_neg.in_features, mlp.c_proj_pos_pos.in_features)
        self.assertEqual(mlp.c_proj_pos_neg.in_features, mlp.c_proj_neg_pos.in_features)


if __name__ == "__main__":
    unittest.main()
