"""Persisted sample offsets for paired, architecture-independent evaluation."""
import json
from pathlib import Path
import torch


def create_offset_manifest(path, *, data_length, block_size, batches, batch_size, seed):
    if data_length <= block_size:
        raise ValueError("data must be longer than block_size")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    offsets = torch.randint(data_length - block_size, (batches, batch_size), generator=generator).tolist()
    manifest = {"seed": seed, "data_length": data_length, "block_size": block_size, "offsets": offsets}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_offset_manifest(path):
    return json.loads(Path(path).read_text())
