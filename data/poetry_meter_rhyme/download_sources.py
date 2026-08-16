#!/usr/bin/env python3
"""Download immutable poetry sources described by sources.lock.json."""
import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock-file", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--accept-research-only", action="store_true")
    parser.add_argument("--include-background", action="store_true")
    args = parser.parse_args()
    lock = json.loads(Path(args.lock_file).read_text())
    raw = Path(args.raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    for source in lock["sources"]:
        if source.get("background") and not args.include_background:
            continue
        if source["license_status"] == "research-only" and not args.accept_research_only:
            raise SystemExit(f"{source['name']} annotations require --accept-research-only")
        target = raw / source["name"]
        marker = target / ".complete"
        if marker.exists() and marker.read_text().strip() == source["sha256"]:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "download"
            with urllib.request.urlopen(source["url"]) as response, archive.open("wb") as out:
                shutil.copyfileobj(response, out)
            actual = sha256(archive)
            if actual != source["sha256"]:
                raise SystemExit(f"checksum mismatch for {source['name']}: {actual}")
            shutil.rmtree(target, ignore_errors=True)
            target.mkdir()
            if zipfile.is_zipfile(archive):
                with zipfile.ZipFile(archive) as bundle:
                    bundle.extractall(target)
            elif tarfile.is_tarfile(archive):
                with tarfile.open(archive) as bundle:
                    bundle.extractall(target, filter="data")
            else:
                shutil.copy2(archive, target / source.get("filename", "source.data"))
            marker.write_text(source["sha256"] + "\n")


if __name__ == "__main__":
    main()
