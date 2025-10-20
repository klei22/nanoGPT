#!/usr/bin/env python3
"""Push ExecuTorch artifacts to an Android device via ADB."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

DEFAULT_DEVICE_DIR = "/sdcard/Android/data/com.example.edgeai/files/models"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        required=True,
        help="Directory produced by export_to_executorch.py",
    )
    parser.add_argument(
        "--adb",
        type=str,
        default="adb",
        help="ADB executable to invoke (defaults to the system PATH entry).",
    )
    parser.add_argument(
        "--device-dir",
        type=str,
        default=DEFAULT_DEVICE_DIR,
        help="Target directory on the device for model artifacts.",
    )
    parser.add_argument(
        "--clean-remote",
        action="store_true",
        help="Delete the remote directory before pushing new files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned ADB commands without executing them.",
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


def _run_adb(adb: str, args: Iterable[str], dry_run: bool) -> None:
    command = [adb, *args]
    if dry_run:
        logging.info("[dry-run] Would run: %s", " ".join(command))
        return
    logging.debug("Running: %s", " ".join(command))
    subprocess.run(command, check=True)


def _check_adb(adb: str, dry_run: bool) -> None:
    if dry_run:
        if shutil.which(adb) is None:
            logging.warning("ADB binary '%s' not found on PATH (dry-run mode).", adb)
        return
    if shutil.which(adb) is None:
        raise FileNotFoundError(
            f"ADB binary '{adb}' not found. Install Android platform tools or provide --adb."
        )

    subprocess.run([adb, "start-server"], check=True)
    subprocess.run([adb, "devices"], check=True)


def _iter_sources(artifacts: Path) -> Iterable[Path]:
    for path in sorted(artifacts.iterdir()):
        if path.name.startswith("."):
            continue
        yield path


def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbose)

    artifacts = args.artifacts.expanduser().resolve()
    if not artifacts.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts}")

    _check_adb(args.adb, args.dry_run)

    sources = list(_iter_sources(artifacts))
    if not sources:
        logging.warning("No files found in %s to push", artifacts)

    if args.clean_remote:
        _run_adb(args.adb, ["shell", "rm", "-rf", args.device_dir], args.dry_run)

    _run_adb(args.adb, ["shell", "mkdir", "-p", args.device_dir], args.dry_run)

    for source in sources:
        remote_path = f"{args.device_dir.rstrip('/')}/{source.name}"
        _run_adb(args.adb, ["push", str(source), remote_path], args.dry_run)

    logging.info("Pushed %d artifact(s) to %s", len(sources), args.device_dir)


if __name__ == "__main__":
    main()
