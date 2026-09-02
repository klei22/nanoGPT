"""Readout primitives for controlled predictive-width experiments.

These modules deliberately operate on ``[B,T,S,q]`` state.  In particular the
collapsed control still executes every transform and every vocabulary head;
only the predictive cut differs.
"""
import math
import torch
import torch.nn as nn


class StreamReadout(nn.Module):
    def __init__(self, n_streams, stream_dim, vocab_size, mode="direct", bias=False):
        super().__init__()
        if mode not in {"direct", "collapsed"}:
            raise ValueError(f"unsupported stream readout: {mode}")
        self.n_streams, self.stream_dim, self.mode = n_streams, stream_dim, mode
        self.transforms = nn.ModuleList([nn.Linear(stream_dim, stream_dim, bias=bias) for _ in range(n_streams)])
        self.heads = nn.ModuleList([nn.Linear(stream_dim, vocab_size, bias=False) for _ in range(n_streams)])

    def tie_heads(self, embeddings):
        if len(embeddings) != self.n_streams:
            raise ValueError("one embedding is required for every stream")
        for head, embedding in zip(self.heads, embeddings):
            head.weight = embedding.weight

    def path_logits(self, states):
        if states.ndim != 4 or states.shape[-2:] != (self.n_streams, self.stream_dim):
            raise ValueError("states must have shape [B,T,S,q]")
        transformed = [layer(states[:, :, i]) for i, layer in enumerate(self.transforms)]
        if self.mode == "collapsed":
            collapsed = torch.stack(transformed, dim=0).sum(0) / math.sqrt(self.n_streams)
            inputs = [collapsed] * self.n_streams
        else:
            inputs = transformed
        return torch.stack([head(value) for head, value in zip(self.heads, inputs)], dim=-2)

    def forward(self, states):
        return self.path_logits(states).sum(dim=-2) / math.sqrt(self.n_streams)


class LateNonlinearReadout(nn.Module):
    def __init__(self, model_dim, tap_dim, vocab_size, nonlinear=True, init_scale=0.01):
        super().__init__()
        self.up = nn.Linear(model_dim, tap_dim, bias=False)
        self.head = nn.Linear(tap_dim, vocab_size, bias=False)
        self.nonlinear = nonlinear
        self.alpha = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, state):
        feature = self.up(state)
        if self.nonlinear:
            feature = torch.relu(feature).square()
        return self.alpha * self.head(feature)


DirectStreamReadout = StreamReadout
CollapsedStreamReadout = StreamReadout
