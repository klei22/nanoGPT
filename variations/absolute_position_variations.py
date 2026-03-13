import torch
import torch.nn as nn


class CyclicAbsolutePositionEmbedding(nn.Module):
    """Absolute position embeddings formed by summing multiple cyclic channels.

    Each channel has its own period. For token position p, channel i selects
    index (p + offset_i) % period_i. All selected vectors are summed.
    """

    def __init__(self, periods, n_embd, random_start=True):
        super().__init__()
        if not periods:
            raise ValueError("periods must be a non-empty list of positive integers")
        if any((not isinstance(period, int)) or period <= 0 for period in periods):
            raise ValueError("all periods must be positive integers")

        self.periods = list(periods)
        self.channels = nn.ModuleList([nn.Embedding(period, n_embd) for period in self.periods])

        if random_start:
            starts = [torch.randint(low=0, high=period, size=(1,)).item() for period in self.periods]
        else:
            starts = [0 for _ in self.periods]
        self.register_buffer("start_offsets", torch.tensor(starts, dtype=torch.long), persistent=True)

    def forward(self, pos):
        out = None
        for channel_idx, (period, embedding) in enumerate(zip(self.periods, self.channels)):
            offset_pos = (pos + self.start_offsets[channel_idx]) % period
            channel_out = embedding(offset_pos)
            out = channel_out if out is None else out + channel_out
        return out

