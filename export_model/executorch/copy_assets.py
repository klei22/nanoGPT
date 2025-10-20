#!/usr/bin/env python3
"""Copy exported ExecuTorch artifacts into the Android app assets."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Iterable

DEFAULT_APP_ROOT = Path(__file__).resolve().parents[2] / "EdgeAIApp-ExecuTorch"
ASSETS_SUBDIR = Path("app/src/main/assets/models")
DEFAULT_APP_PACKAGE_FILE = "llama_model.pte"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        required=True,
        help="Directory produced by export_to_executorch.py",
    )
    parser.add_argument(
        "--app",
        type=Path,
        default=DEFAULT_APP_ROOT,
        help="Path to the EdgeAIApp-ExecuTorch repository root.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Name of the folder created inside assets/models/.",
    )
    parser.add_argument(
        "--clear-target",
        action="store_true",
        help="Remove the destination folder before copying new assets.",
    )
    parser.add_argument(
        "--install-default-name",
        action="store_true",
        help=(
            "Also copy the .pte file as 'llama_model.pte' in assets/models"
            " to match the current Kotlin loaders."
        ),
    )
    parser.add_argument(
        "--overwrite-default",
        action="store_true",
        help="Allow replacing an existing default llama_model.pte file.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions only.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=level)


def _detect_model_name(artifacts: Path, override: str | None) -> tuple[str, Path]:
    if override:
        pte_candidates = list(artifacts.glob("*.pte"))
        if not pte_candidates:
            raise FileNotFoundError("No .pte file found in artifacts directory")
        return override, pte_candidates[0]

    pte_files = list(artifacts.glob("*.pte"))
    if not pte_files:
        raise FileNotFoundError("No .pte file found in artifacts directory")
    if len(pte_files) > 1:
        raise ValueError("Multiple .pte files found. Use --model-name to disambiguate.")
    return pte_files[0].stem, pte_files[0]


def _iter_sources(artifacts: Path) -> Iterable[Path]:
    for path in sorted(artifacts.iterdir()):
        if path.name.startswith("."):
            continue
        yield path


def _copy_path(src: Path, dest: Path, dry_run: bool) -> None:
    if dry_run:
        logging.info("[dry-run] Would copy %s -> %s", src, dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
    else:
        shutil.copy2(src, dest)


def _clear_target(target: Path, dry_run: bool) -> None:
    if dry_run:
        logging.info("[dry-run] Would remove %s", target)
        return
    if target.exists():
        shutil.rmtree(target)


def _install_default(pte_src: Path, assets_root: Path, overwrite: bool, dry_run: bool) -> None:
    default_path = assets_root / DEFAULT_APP_PACKAGE_FILE
    if default_path.exists() and not overwrite and not dry_run:
        raise FileExistsError(
            f"Default model already exists at {default_path}. Use --overwrite-default to replace it."
        )
    _copy_path(pte_src, default_path, dry_run=dry_run)


def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbose)

    artifacts = args.artifacts.expanduser().resolve()
    if not artifacts.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts}")

    app_root = args.app.expanduser().resolve()
    assets_root = app_root / ASSETS_SUBDIR
    if not assets_root.exists():
        raise FileNotFoundError(
            f"Android assets directory not found at {assets_root}."
            " Verify the --app argument points to EdgeAIApp-ExecuTorch."
        )

    model_name, pte_file = _detect_model_name(artifacts, args.model_name)
    logging.info("Installing artifacts for model '%s'", model_name)

    target_dir = assets_root / model_name
    if args.clear_target:
        _clear_target(target_dir, args.dry_run)

    for source in _iter_sources(artifacts):
        destination = target_dir / source.name
        _copy_path(source, destination, dry_run=args.dry_run)

    if args.install_default_name:
        _install_default(pte_file, assets_root, args.overwrite_default, dry_run=args.dry_run)

    logging.info("Assets copied to %s", target_dir)


if __name__ == "__main__":
    main()
