import json

import torch
import pytest

from analysis.dual_stream_clock import (batches, evaluation_batch, main, make_config,
                                       parse_args, snapshot)
from gpt_conf import GPTConfig
from model import GPT


@pytest.mark.parametrize("variant,radius_fraction", [("small_circle_r10", .10), ("small_circle_r05", .05)])
def test_both_streams_start_on_independent_learnable_tiny_circles(variant, radius_fraction):
    args = parse_args(["--variants", variant])
    model = GPT(make_config(args, variant))
    digit, letter = model.transformer.wte_0, model.transformer.wte_1
    assert digit.raw_frame is not letter.raw_frame
    for embedding in (digit, letter):
        geometry = embedding.geometry()
        fraction = (geometry["radius"] / geometry["sphere_radius"]).detach().item()
        assert fraction == pytest.approx(radius_fraction, abs=2e-6)
        assert embedding.offset_logit.requires_grad
        assert torch.allclose(embedding.weight.norm(dim=-1), torch.full((embedding.num_embeddings,), args.radius), atol=2e-6)
        embedding.embed_phase(torch.tensor([.1, .3])).square().sum(dim=0)[0].backward()
        assert torch.isfinite(embedding.offset_logit.grad)


def test_paired_next_token_targets_and_complete_default_cycle():
    args = parse_args([])
    inputs, targets = batches(torch.tensor([0, 7]), torch.tensor([0, 4]), args)
    assert inputs["digits"][1, 0] == 7 and targets["digits"][1, 0] == 0
    assert inputs["letters"][1, 0] == 4 and targets["letters"][1, 0] == 0
    eval_inputs, _ = evaluation_batch(args)
    pairs = set(zip(eval_inputs["digits"][:, 0].tolist(), eval_inputs["letters"][:, 0].tolist()))
    assert len(pairs) == 8 * 5
    assert args.digit_slots == 10  # 8 and 9 remain competing, untargeted classes


def test_checkpoint_snapshot_alignment_and_exact_resume(tmp_path):
    common = ["--variants", "small_circle", "--block-size", "4", "--batch-size", "2",
              "--checkpoint-every", "2", "--log-every", "10"]
    first = common + ["--output-dir", str(tmp_path / "resumed"), "--checkpoint-dir", str(tmp_path / "ckpt")]
    main(first + ["--steps", "2"])
    main(first + ["--steps", "4", "--resume"])
    main(common + ["--steps", "4", "--output-dir", str(tmp_path / "full"), "--checkpoint-dir", str(tmp_path / "fullckpt")])
    resumed = json.loads((tmp_path / "resumed/small_circle-seed-0.json").read_text())
    full = json.loads((tmp_path / "full/small_circle-seed-0.json").read_text())
    assert [f["iteration"] for f in resumed["frames"]] == list(range(5))
    assert resumed["frames"] == full["frames"]
    assert resumed["training"] == full["training"]
    saved = torch.load(tmp_path / "ckpt/small_circle-seed-0.pt", weights_only=False)
    model = GPT(GPTConfig(**saved["model_args"]))
    model.load_state_dict(saved["model"])
    args = parse_args(first + ["--steps", "4"])
    assert snapshot(model, args, 4, evaluation_batch(args)) == resumed["frames"][-1]
    assert resumed["continuous_probe"] is not None
    assert all(not key.startswith("val_") for key in resumed["frames"][-1]["metrics"])
