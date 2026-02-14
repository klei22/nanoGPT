#!/usr/bin/env python3
"""Export a trained nanoGPT checkpoint to an ExecuTorch PTE for Android (XNNPACK)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import torch
from torch.export import export
from torch.nn.attention import sdpa_kernel, SDPBackend

from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
from executorch.backends.xnnpack.utils.configs import (
    get_xnnpack_edge_compile_config,
)
from executorch.exir import to_edge_transform_and_lower

from gpt_conf import GPTConfig
from model import GPT


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_TOKENIZER_PATH = (
    REPO_ROOT / "data" / "template" / "programming_tokenizers" / "python_tokenizer.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a nanoGPT checkpoint to an ExecuTorch PTE for Android."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("out") / "ckpt.pt",
        help="Path to the trained checkpoint (ckpt.pt).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("exutorch") / "android_export",
        help="Directory to write the PTE and tokenizer assets.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Optional override for model block size used during export.",
    )
    parser.add_argument(
        "--tokenizer-meta",
        type=Path,
        default=None,
        help="Optional path to a meta.pkl for tokenizer export.",
    )
    return parser.parse_args()


def load_checkpoint(checkpoint_path: Path) -> tuple[GPT, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_args = checkpoint.get("model_args")
    if model_args is None:
        raise ValueError("Checkpoint is missing model_args; cannot reconstruct GPTConfig.")

    model_args = dict(model_args)
    model_args["dropout"] = 0.0
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, checkpoint


def resolve_meta_paths(
    checkpoint: dict, checkpoint_path: Path, explicit_meta: Path | None
) -> dict[str, Path]:
    if explicit_meta is not None:
        return {"default": explicit_meta}

    checkpoint_config = checkpoint.get("config", {}) or {}
    datasets = checkpoint_config.get("multicontext_datasets")
    if datasets:
        return {
            dataset: REPO_ROOT / "data" / dataset / "meta.pkl" for dataset in datasets
        }

    ckpt_dir_meta = checkpoint_path.parent / "meta.pkl"
    if ckpt_dir_meta.exists():
        return {"default": ckpt_dir_meta}

    dataset = checkpoint_config.get("dataset")
    if dataset:
        return {"default": REPO_ROOT / "data" / dataset / "meta.pkl"}

    return {}


def export_tokenizer_assets(meta_paths: dict[str, Path], out_dir: Path) -> list[Path]:
    tokenizer_dir = out_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    exported = []

    for name, meta_path in meta_paths.items():
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.pkl not found at {meta_path}")
        with meta_path.open("rb") as f:
            meta = pickle.load(f)

        if name == "default":
            dest_meta = tokenizer_dir / "meta.pkl"
        else:
            dest_meta = tokenizer_dir / name / "meta.pkl"
            dest_meta.parent.mkdir(parents=True, exist_ok=True)

        dest_meta.write_bytes(meta_path.read_bytes())
        exported.append(dest_meta)

        tokenizer_type = meta.get("tokenizer")
        if tokenizer_type == "python_json_byte_fallback":
            if not PYTHON_TOKENIZER_PATH.exists():
                raise FileNotFoundError(
                    f"python_tokenizer.py not found at {PYTHON_TOKENIZER_PATH}"
                )
            tokenizer_dest = dest_meta.parent / "python_tokenizer.py"
            tokenizer_dest.write_bytes(PYTHON_TOKENIZER_PATH.read_bytes())
            exported.append(tokenizer_dest)

    return exported


def export_pte(model: GPT, out_dir: Path, block_size_override: int | None) -> Path:
    if block_size_override is not None:
        model.update_block_size(block_size_override)

    vocab_size = model.config.vocab_size
    block_size = model.config.block_size
    example_inputs = (torch.randint(0, vocab_size, (1, block_size), dtype=torch.long),)
    dynamic_shape = (
        {1: torch.export.Dim("token_dim", max=block_size)},
    )

    with sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
        traced_model = export(model, example_inputs, dynamic_shapes=dynamic_shape)

    edge_config = get_xnnpack_edge_compile_config()
    edge_manager = to_edge_transform_and_lower(
        traced_model,
        partitioner=[XnnpackPartitioner()],
        compile_config=edge_config,
    )
    et_program = edge_manager.to_executorch()

    out_dir.mkdir(parents=True, exist_ok=True)
    pte_path = out_dir / "nanogpt_xnnpack.pte"
    pte_path.write_bytes(et_program.buffer)
    return pte_path


def main() -> None:
    args = parse_args()
    model, checkpoint = load_checkpoint(args.checkpoint)

    meta_paths = resolve_meta_paths(checkpoint, args.checkpoint, args.tokenizer_meta)
    if not meta_paths:
        raise FileNotFoundError(
            "No meta.pkl found. Provide --tokenizer-meta or ensure training copied it."
        )

    pte_path = export_pte(model, args.out_dir, args.block_size)
    tokenizer_files = export_tokenizer_assets(meta_paths, args.out_dir)

    manifest = {
        "checkpoint": str(args.checkpoint),
        "pte_file": pte_path.name,
        "tokenizer_files": [str(path.relative_to(args.out_dir)) for path in tokenizer_files],
        "model_config": checkpoint.get("config", {}),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Wrote ExecuTorch program to: {pte_path}")
    print("Tokenizer assets:")
    for path in tokenizer_files:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
