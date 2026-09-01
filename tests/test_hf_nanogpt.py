import math
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from hf_model import NanoGPTConfig, NanoGPTForCausalLM
from hf_model.modeling_nanogpt import _apply_rope
from scripts.compare_hf_relu2max import evaluation_strategy_argument


def test_direct_comparison_script_can_import_local_model():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "compare_hf_relu2max.py"), "--help"],
        cwd=repo_root.parent,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ReLU2Max/softmax" in result.stdout


def test_training_arguments_accept_selected_evaluation_keyword(tmp_path):
    from transformers import TrainingArguments

    arguments = TrainingArguments(
        output_dir=str(tmp_path),
        report_to=[],
        **evaluation_strategy_argument(),
    )
    strategy = getattr(arguments, "eval_strategy", None)
    if strategy is None:
        strategy = arguments.evaluation_strategy
    assert strategy.value == "steps"


def tiny_config(**kwargs):
    values = dict(vocab_size=31, max_position_embeddings=16, hidden_size=16,
                  intermediate_size=32, num_hidden_layers=2,
                  num_attention_heads=4, rope_length=4)
    values.update(kwargs)
    return NanoGPTConfig(**values)


@pytest.mark.parametrize("normalizer", ["relu2max", "softmax"])
def test_causal_lm_forward_backward_and_roundtrip(tmp_path, normalizer):
    model = NanoGPTForCausalLM(tiny_config(attention_normalizer=normalizer))
    input_ids = torch.randint(0, 31, (2, 8))
    output = model(input_ids, labels=input_ids)
    assert output.logits.shape == (2, 8, 31)
    assert torch.isfinite(output.loss)
    output.loss.backward()

    model.save_pretrained(tmp_path)
    loaded = NanoGPTForCausalLM.from_pretrained(tmp_path).eval()
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(input_ids).logits, loaded(input_ids).logits)


def test_qk_scale_and_partial_interleaved_rope():
    config = tiny_config()
    assert config.qk_norm_scale_init == pytest.approx(math.log2(16 * 16 - 16))
    x = torch.tensor([[[[1.0, 0.0, 3.0, 4.0, 9.0, 10.0]]]])
    result = _apply_rope(x, torch.tensor([[1]]), 4, 10000.0)
    torch.testing.assert_close(result[..., 4:], x[..., 4:])
    torch.testing.assert_close(result[..., 0], torch.cos(torch.tensor(1.0)))
    torch.testing.assert_close(result[..., 1], torch.sin(torch.tensor(1.0)))


def test_generation_cache_matches_full_forward():
    model = NanoGPTForCausalLM(tiny_config(attention_normalizer="relu2max")).eval()
    ids = torch.randint(0, 31, (1, 6))
    with torch.no_grad():
        full = model(ids).logits[:, -1]
        prefix = model(ids[:, :-1], use_cache=True)
        cached = model(ids[:, -1:], past_key_values=prefix.past_key_values, use_cache=True).logits[:, -1]
    torch.testing.assert_close(full, cached, atol=1e-5, rtol=1e-5)
