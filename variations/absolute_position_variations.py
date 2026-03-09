# variations/absolute_position_variations.py
"""
Absolute position embedding variations.

Provides alternatives to the standard single learned absolute position embedding:

- "standard": The default nn.Embedding(block_size, n_embd) — backwards compatible.
- "multi_channel_cyclic": Creates multiple smaller embedding tables (channels),
  each with a different cycle length. At each position, the embedding from each
  channel is looked up (cycling), and all channels are summed together. An
  optional randomized start index per channel breaks symmetry and may improve
  length extrapolation.
"""

import torch
import torch.nn as nn
import random


class MultiChannelCyclicPositionEmbedding(nn.Module):
    """Multiple cycling embedding channels summed together.

    Each channel *i* has an embedding table of size ``cycle_lengths[i]`` by
    ``n_embd``.  For a given sequence position *t* the lookup index into
    channel *i* is::

        (t + start_offsets[i]) % cycle_lengths[i]

    All channel outputs are summed to produce the final positional embedding.

    Parameters
    ----------
    cycle_lengths : list[int]
        Number of unique embedding vectors per channel.
    n_embd : int
        Embedding dimension (must match the model width).
    random_start : bool
        If ``True`` (default) each channel receives a random start offset
        drawn uniformly in ``[0, cycle_length)``.  This is set once at init
        and stored as a buffer so it is saved/loaded with the checkpoint.
    seed : int or None
        Optional seed for the random start offsets (for reproducibility).
    """

    def __init__(self, cycle_lengths, n_embd, random_start=True, seed=None):
        super().__init__()
        self.cycle_lengths = list(cycle_lengths)
        self.n_embd = n_embd
        self.n_channels = len(self.cycle_lengths)

        # One embedding table per channel
        self.embeddings = nn.ModuleList([
            nn.Embedding(cl, n_embd) for cl in self.cycle_lengths
        ])

        # Compute (deterministic or random) start offsets
        if random_start:
            rng = random.Random(seed)
            offsets = [rng.randint(0, cl - 1) for cl in self.cycle_lengths]
        else:
            offsets = [0] * self.n_channels

        self.register_buffer(
            "start_offsets",
            torch.tensor(offsets, dtype=torch.long),
        )

    def forward(self, pos):
        """
        Parameters
        ----------
        pos : torch.Tensor
            1-D position indices of shape ``(T,)``.

        Returns
        -------
        torch.Tensor
            Positional embeddings of shape ``(T, n_embd)``.
        """
        out = torch.zeros(pos.size(0), self.n_embd, device=pos.device, dtype=self.embeddings[0].weight.dtype)
        for i, emb in enumerate(self.embeddings):
            idx = (pos + self.start_offsets[i]) % self.cycle_lengths[i]
            out = out + emb(idx)
        return out


# ---------------------------------------------------------------------------
# Factory / registry
# ---------------------------------------------------------------------------

def build_absolute_position_embedding(config):
    """Return the position-embedding module selected by ``config.abs_pos_variant``.

    Returns
    -------
    nn.Module
        A module whose ``forward(pos)`` accepts a 1-D LongTensor of position
        indices and returns ``(T, n_embd)``.
    """
    variant = getattr(config, "abs_pos_variant", "standard")

    if variant == "standard":
        # Backwards-compatible: plain learned table
        return nn.Embedding(config.block_size, config.n_embd)

    if variant == "multi_channel_cyclic":
        cycle_lengths = getattr(config, "abs_pos_cycle_lengths", None)
        if cycle_lengths is None or len(cycle_lengths) == 0:
            raise ValueError(
                "abs_pos_variant='multi_channel_cyclic' requires "
                "abs_pos_cycle_lengths to be set (list of ints)."
            )
        random_start = getattr(config, "abs_pos_random_start", True)
        seed = getattr(config, "abs_pos_random_seed", None)
        return MultiChannelCyclicPositionEmbedding(
            cycle_lengths=cycle_lengths,
            n_embd=config.n_embd,
            random_start=random_start,
            seed=seed,
        )

    raise ValueError(f"Unknown abs_pos_variant: {variant!r}")
