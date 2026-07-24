import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from model import GPT


class ImportWteLmHeadTest(unittest.TestCase):
    def test_normalizes_each_lm_head_row_before_scaling(self):
        model = GPT.__new__(GPT)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(multicontext=False, multidataset_wte=False)
        model.uses_numerical_multicontext = False
        model.wte_weight_tying = True
        model.transformer = nn.ModuleDict({"wte": nn.Embedding(3, 2)})
        model.lm_head = nn.Linear(2, 3, bias=False)
        model.lm_head.weight = model.transformer.wte.weight

        imported_weight = torch.tensor([[3.0, 4.0], [0.0, 2.0], [-5.0, 0.0]])
        checkpoint = {
            "model": {
                "transformer.wte.weight": imported_weight,
                "lm_head.weight": imported_weight.clone(),
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "ckpt.pt"
            torch.save(checkpoint, checkpoint_path)
            model.import_wte_lm_head_from_ckpt(
                checkpoint_path, normalize_lm_head=True, lm_head_scale=0.8
            )

        torch.testing.assert_close(model.transformer.wte.weight, imported_weight)
        torch.testing.assert_close(
            model.lm_head.weight.norm(dim=1), torch.full((3,), 0.8)
        )
        self.assertIsNot(model.lm_head.weight, model.transformer.wte.weight)
        self.assertFalse(model.wte_weight_tying)


if __name__ == "__main__":
    unittest.main()
