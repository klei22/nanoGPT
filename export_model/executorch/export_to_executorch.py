#!/usr/bin/env python3
"""Export nanoGPT checkpoints into ExecuTorch programs.

The generated `.pte` file plus metadata can be consumed by the
EdgeAIApp-ExecuTorch Android demo.  See the module level README for detailed
usage instructions.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch

from gpt_conf import GPTConfig
from model import GPT

try:  # Lazy import so the script still shows a friendly error when missing
    from executorch.exir import EdgeCompileConfig, to_edge
except ImportError as exc:  # pragma: no cover - error message path
    raise SystemExit(
        "ExecuTorch is required for export. Install it from"
        " https://pytorch.org/executorch/ before running this script."
    ) from exc

try:
    from torch.export import export, export_for_training
except ImportError as exc:  # pragma: no cover - torch is too old
    raise SystemExit(
        "This script requires torch.export (PyTorch 2.1+)."
        " Upgrade PyTorch to continue."
    ) from exc


def _sdpa_context():
    """Return a context manager that forces math SDPA backend if available."""

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception:  # pragma: no cover - fallback on older PyTorch
        return contextlib.nullcontext()
    return sdpa_kernel([SDPBackend.MATH])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the nanoGPT training checkpoint (.pt/.pth).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("export_artifacts"),
        help="Directory for the ExecuTorch program and metadata.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Base name for generated files. Defaults to the checkpoint stem.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help=(
            "Override the context length used for tracing.  Useful when the"
            " Android runtime should accept fewer tokens than the training"
            " checkpoint."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16"],
        default="float32",
        help="Convert model weights to this dtype before export.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Optional tokenizer file to copy beside the exported program.",
    )
    parser.add_argument(
        "--extra-asset",
        type=Path,
        action="append",
        default=None,
        help="Additional files to copy into the output directory.",
    )
    parser.add_argument(
        "--check-ir",
        action="store_true",
        help="Run ExecuTorch IR validation checks during compilation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without performing export.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output.",
    )
    return parser.parse_args()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=level)


def _resolve_model_name(checkpoint: Path, override: str | None) -> str:
    if override:
        return override
    return checkpoint.stem


def _load_checkpoint(checkpoint: Path) -> dict:
    logging.info("Loading checkpoint from %s", checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return torch.load(checkpoint, map_location="cpu")


def _build_model(ckpt: dict, dtype: torch.dtype, block_size_override: int | None) -> tuple[GPT, GPTConfig]:
    model_args = dict(ckpt["model_args"])
    if block_size_override is not None:
        logging.info("Overriding block_size -> %s", block_size_override)
        model_args["block_size"] = block_size_override
    config = GPTConfig(**model_args)
    model = GPT(config)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    model.to(dtype=dtype)
    return model, config


def _copy_assets(paths: Iterable[Path], destination: Path, dry_run: bool) -> None:
    for path in paths:
        if not path:
            continue
        target = destination / path.name
        if dry_run:
            logging.info("[dry-run] Would copy %s -> %s", path, target)
            continue
        logging.info("Copying %s -> %s", path, target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)


def _write_metadata(
    output_dir: Path,
    model_name: str,
    checkpoint: Path,
    config: GPTConfig,
    block_size: int,
    dtype: str,
    dry_run: bool,
) -> None:
    metadata_path = output_dir / f"{model_name}.json"
    metadata = {
        "model_name": model_name,
        "source_checkpoint": str(checkpoint.resolve()),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dtype": dtype,
        "block_size": block_size,
        "vocab_size": config.vocab_size,
        "config": asdict(config),
    }
    if dry_run:
        logging.info("[dry-run] Would write metadata to %s", metadata_path)
        return
    logging.info("Writing metadata -> %s", metadata_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2))


def _export_to_executorch(
    model: GPT,
    config: GPTConfig,
    model_name: str,
    output_dir: Path,
    block_size: int,
    check_ir: bool,
    dry_run: bool,
) -> Path:
    pte_path = output_dir / f"{model_name}.pte"
    if dry_run:
        logging.info("[dry-run] Would export ExecuTorch program -> %s", pte_path)
        return pte_path

    example_inputs = (
        torch.randint(0, config.vocab_size, (1, block_size), dtype=torch.long),
    )
    dynamic_shapes = (
        {1: torch.export.Dim("token_dim", max=block_size)},
    )

    logging.info("Tracing model for ExecuTorch export (block_size=%s)", block_size)
    with _sdpa_context(), torch.no_grad():
        exported_module = export_for_training(
            model, example_inputs, dynamic_shapes=dynamic_shapes
        ).module()
        traced_module = export(
            exported_module, example_inputs, dynamic_shapes=dynamic_shapes
        )

    logging.info("Compiling ExecuTorch program")
    edge_config = EdgeCompileConfig(_check_ir_validity=check_ir)
    edge_manager = to_edge(traced_module, compile_config=edge_config)
    et_program = edge_manager.to_executorch()

    logging.info("Writing ExecuTorch program -> %s", pte_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    with pte_path.open("wb") as file:
        file.write(et_program.buffer)
    return pte_path


def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbose)

    model_name = _resolve_model_name(args.checkpoint, args.model_name)
    logging.info("Model name resolved to '%s'", model_name)

    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    ckpt = _load_checkpoint(args.checkpoint)
    model, config = _build_model(ckpt, dtype=dtype, block_size_override=args.block_size)
    block_size = args.block_size or config.block_size

    pte_path = _export_to_executorch(
        model,
        config,
        model_name,
        args.output_dir,
        block_size,
        check_ir=args.check_ir,
        dry_run=args.dry_run,
    )

    _write_metadata(
        args.output_dir,
        model_name,
        args.checkpoint,
        config,
        block_size,
        args.dtype,
        dry_run=args.dry_run,
    )

    assets = []
    if args.tokenizer:
        assets.append(args.tokenizer)
    if args.extra_asset:
        assets.extend(args.extra_asset)
    if assets:
        _copy_assets(assets, args.output_dir, args.dry_run)

    logging.info("Export complete. ExecuTorch program at %s", pte_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
