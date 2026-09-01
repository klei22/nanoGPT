"""Train a Hugging Face-compatible nanoGPT with Full Attention Residuals.

The model keeps the repository's default architectural choices (RoPE, QK RMS
normalisation with a learned per-head scale, pre-norm, and squared ReLU MLPs),
while making the depth router selectable between softmax and normalised ReLU.

Example::

    python huggingface_model/train_attention_residual.py \
        --dataset_name wikitext --dataset_config wikitext-2-raw-v1 \
        --tokenizer_name gpt2 --router_activation softmax

The resulting directory is loadable with ``AttentionResidualForCausalLM``'s
``from_pretrained`` method.  This is intentionally a training script rather
than a modification of a pretrained architecture: Full Attention Residuals
change the dataflow between every decoder sublayer.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PretrainedConfig,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import CausalLMOutput

from train_variations.muon import SingleDeviceMuonWithAuxAdam


class AttentionResidualConfig(PretrainedConfig):
    """Configuration with nanoGPT-sized defaults and HF serialization."""

    model_type = "nanogpt_attention_residual"

    def __init__(
        self,
        vocab_size: int = 50304,
        block_size: int = 1024,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        rms_norm_eps: float = 1e-6,
        qk_norm_scale_init: Optional[float] = None,
        router_activation: str = "softmax",
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        if n_embd % n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if router_activation not in {"softmax", "relu"}:
            raise ValueError("router_activation must be 'softmax' or 'relu'")
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.max_position_embeddings = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.rope_base = rope_base
        self.rms_norm_eps = rms_norm_eps
        # Match nanoGPT's QK-norm scale initialization.
        self.qk_norm_scale_init = qk_norm_scale_init or math.log2(block_size * block_size - block_size)
        self.router_activation = router_activation


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float) -> torch.Tensor:
    """Apply rotary embeddings to ``(batch, heads, time, head_dim)``."""
    rotary_width = x.size(-1) - x.size(-1) % 2
    inv_freq = base ** (-torch.arange(0, rotary_width, 2, device=x.device).float() / rotary_width)
    angles = torch.outer(positions.float(), inv_freq).to(x.dtype)
    cos, sin = angles.cos()[None, None], angles.sin()[None, None]
    x_rot, x_tail = x[..., :rotary_width], x[..., rotary_width:]
    even, odd = x_rot[..., ::2], x_rot[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)
    return torch.cat((rotated, x_tail), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: AttentionResidualConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.rope_base = config.rope_base
        self.eps = config.rms_norm_eps
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.qk_scale = nn.Parameter(torch.tensor(config.qk_norm_scale_init))
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        reshape = lambda t: t.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)
        positions = torch.arange(length, device=x.device)
        q, k = apply_rope(q, positions, self.rope_base), apply_rope(k, positions, self.rope_base)
        q = F.normalize(q, dim=-1, eps=self.eps)
        k = F.normalize(k, dim=-1, eps=self.eps)
        scores = (q @ k.transpose(-2, -1)) * self.qk_scale
        causal = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
        allowed = causal[None, None]
        if attention_mask is not None:
            allowed = allowed & attention_mask[:, None, None, :].bool()
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype)
        weights = F.dropout(weights, self.dropout, self.training)
        return self.proj((weights @ v).transpose(1, 2).contiguous().view(batch, length, width))


class SquaredReLUMLP(nn.Module):
    def __init__(self, config: AttentionResidualConfig):
        super().__init__()
        hidden = config.mlp_ratio * config.n_embd
        self.up = nn.Linear(config.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.relu(self.up(x)).square())


class DecoderLayer(nn.Module):
    def __init__(self, config: AttentionResidualConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.mlp_norm = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.mlp = SquaredReLUMLP(config)


class DepthRouter(nn.Module):
    """Token-local Full Attention Residual router over previous sublayers."""

    def __init__(self, destinations: int, width: int, activation: str, eps: float):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(destinations, width))
        self.activation = activation
        self.eps = eps

    def forward(self, sources: list[torch.Tensor], destination: int) -> torch.Tensor:
        values = torch.stack(sources, dim=0)
        keys = F.rms_norm(values, (values.size(-1),), eps=self.eps)
        scores = torch.einsum("dbtc,c->dbt", keys, self.queries[destination])
        if self.activation == "softmax":
            weights = scores.softmax(dim=0)
        else:
            positive = F.relu(scores)
            denominator = positive.sum(dim=0, keepdim=True)
            uniform = torch.full_like(positive, 1.0 / len(sources))
            weights = torch.where(denominator > self.eps, positive / denominator.clamp_min(self.eps), uniform)
        return torch.einsum("dbt,dbtc->btc", weights, values)


class AttentionResidualForCausalLM(PreTrainedModel):
    config_class = AttentionResidualConfig
    base_model_prefix = "transformer"
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: AttentionResidualConfig):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layer)])
        self.router = DepthRouter(2 * config.n_layer + 1, config.n_embd, config.router_activation, config.rms_norm_eps)
        self.norm = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.post_init()
        if config.tie_word_embeddings:
            self.tie_weights()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, value):
        self.wte = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(self, input_ids, attention_mask=None, labels=None, **_) -> CausalLMOutput:
        sources = [self.drop(self.wte(input_ids))]
        destination = 0
        for layer in self.layers:
            x = self.router(sources, destination)
            sources.append(layer.attn(layer.attn_norm(x), attention_mask))
            destination += 1
            x = self.router(sources, destination)
            sources.append(layer.mlp(layer.mlp_norm(x)))
            destination += 1
        logits = self.lm_head(self.norm(self.router(sources, destination)))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)
        return CausalLMOutput(loss=loss, logits=logits)


class MuonTrainer(Trainer):
    """Route decoder matrices to Muon and gains/embeddings/head to AdamW."""

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        adam_names = {"wte.weight", "lm_head.weight"}
        muon, adam = [], []
        for name, parameter in self.model.named_parameters():
            # Scalars/vectors include RMSNorm gains and QK scales. Muon is only
            # defined here for matrix parameters; tied WTE/head is explicitly
            # kept on AdamW regardless of which alias HF exposes.
            if parameter.ndim != 2 or name in adam_names or parameter is self.model.wte.weight:
                adam.append(parameter)
            else:
                muon.append(parameter)
        self.optimizer = SingleDeviceMuonWithAuxAdam([
            {"params": muon, "use_muon": True, "lr": self.args.learning_rate, "momentum": 0.95, "ns_steps": 5},
            {"params": adam, "use_muon": False, "lr": self.args.learning_rate, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": self.args.weight_decay},
        ])
        return self.optimizer


@dataclass
class ScriptArguments:
    dataset_name: str = "wikitext"
    dataset_config: Optional[str] = "wikitext-2-raw-v1"
    tokenizer_name: str = "gpt2"
    text_column: str = "text"
    output_dir: str = "attention-residual-hf"
    router_activation: str = "softmax"
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


def parse_args() -> ScriptArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ScriptArguments.__dataclass_fields__.values():
        kwargs = {"default": field.default}
        if field.name == "router_activation":
            kwargs["choices"] = ("softmax", "relu")
        kwargs["type"] = type(field.default) if field.default is not None else str
        parser.add_argument(f"--{field.name}", **kwargs)
    return ScriptArguments(**vars(parser.parse_args()))


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset(args.dataset_name, args.dataset_config)

    def tokenize(batch):
        return tokenizer(batch[args.text_column], truncation=True, max_length=args.block_size)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)
    config = AttentionResidualConfig(
        vocab_size=len(tokenizer), block_size=args.block_size, n_layer=args.n_layer,
        n_head=args.n_head, n_embd=args.n_embd, router_activation=args.router_activation,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    model = AttentionResidualForCausalLM(config)
    training_args = TrainingArguments(
        output_dir=args.output_dir, learning_rate=3e-4, weight_decay=0.1,
        per_device_train_batch_size=8, per_device_eval_batch_size=8,
        num_train_epochs=1, logging_steps=10, save_strategy="epoch",
        eval_strategy="epoch" if "validation" in tokenized else "no",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to="none",
    )
    trainer = MuonTrainer(
        model=model, args=training_args, train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
