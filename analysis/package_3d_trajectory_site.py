#!/usr/bin/env python3
"""Package completed 3D trajectory runs as a GitHub Pages static site."""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from update_3d_sweep_manifest import update_manifest


SOURCE_DIR = Path("report/threejs/digits-3d")


def package_site(source_dir: Path, output_dir: Path) -> int:
    runs_dir = source_dir / "runs"
    manifest_path = update_manifest(runs_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dual_dir = source_dir / "dual-stream"
    dual_manifest_path = dual_dir / "manifest.json"
    dual_manifest = (json.loads(dual_manifest_path.read_text(encoding="utf-8"))
                     if dual_manifest_path.exists() else {"runs": []})
    if not manifest["runs"] and not dual_manifest["runs"]:
        raise ValueError(f"no completed trajectory JSON files found in {runs_dir}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "runs").mkdir(parents=True)

    selector = (source_dir / "index.html").read_text(encoding="utf-8")
    viewer = (source_dir / "viewer.html").read_text(encoding="utf-8")
    (output_dir / "index.html").write_text(selector, encoding="utf-8")
    (output_dir / "viewer.html").write_text(viewer, encoding="utf-8")
    (output_dir / ".nojekyll").write_text("", encoding="utf-8")

    shutil.copy2(manifest_path, output_dir / "runs/manifest.json")
    for run in manifest["runs"]:
        source = source_dir / run["file"]
        shutil.copy2(source, output_dir / run["file"])

    # Optional two-stream viewer; legacy packaging still works without these files.
    for filename in ("dual-stream.html", "dual-stream.css", "dual-stream.js"):
        if (source_dir / filename).exists():
            shutil.copy2(source_dir / filename, output_dir / filename)
    if dual_manifest["runs"]:
        (output_dir / "dual-stream").mkdir()
        shutil.copy2(dual_manifest_path, output_dir / "dual-stream/manifest.json")
        for run in dual_manifest["runs"]:
            filename = run["file"]
            if Path(filename).name != filename or not filename.endswith(".json"):
                raise ValueError("dual-stream manifest files must be plain JSON filenames")
            shutil.copy2(dual_dir / filename, output_dir / "dual-stream" / filename)
        for filename in ("summary.json", "summary.csv"):
            if (dual_dir / filename).exists():
                shutil.copy2(dual_dir / filename, output_dir / "dual-stream" / filename)

    readme = """# 3D token trajectory static site

This directory is generated. Serve it locally with `python3 -m http.server -d .`
or publish the directory as a GitHub Pages deployment artifact.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return len(manifest["runs"]) + len(dual_manifest["runs"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("dist/digits-3d-site"))
    args = parser.parse_args()
    count = package_site(args.source_dir, args.output_dir)
    print(f"Packaged {count} completed runs into {args.output_dir}")


if __name__ == "__main__":
    main()
