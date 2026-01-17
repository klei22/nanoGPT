"""Demo pipeline to download OPUS-100 (en-zh) and add Hanzi metadata.

The resulting JSON entries contain the raw Chinese text, English
translation, a pinyin rendering, and two flavors of radical/component
decompositions derived from :mod:`hanzipy`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover - dependency handled at runtime
    raise SystemExit(
        "The `datasets` package is required. Install it with `pip install datasets`."
    ) from exc


def _build_pinyin(text: str, char_meta: Dict[str, Dict]) -> str:
    tokens: List[str] = []
    for ch in text:
        info = char_meta.get(ch, {})
        pinyins = info.get("pinyin") or []
        tokens.append(pinyins[0] if pinyins else ch)
    return " ".join(tokens)


def _build_radicals(text: str, char_meta: Dict[str, Dict], prefer_key: str) -> str:
    tokens: List[str] = []
    for ch in text:
        info = char_meta.get(ch, {})
        if prefer_key == "radical":
            parts = info.get("radical") or []
        else:
            components = info.get("components") or {}
            parts = components.get(prefer_key) or []
        token = "".join(parts) if parts else ch
        tokens.append(token)
    return " ".join(tokens)


def _write_jsonl(entries: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False))
            f.write("\n")


def _run_annotation(input_path: Path, output_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "template" / "utils" / "annotate_hanzi_metadata.py"
    cmd = [
        os.fspath(Path(os.sys.executable)),
        os.fspath(script_path),
        "--input_json",
        os.fspath(input_path),
        "--output_json",
        os.fspath(output_path),
        "--text_key",
        "zh",
        "--metadata_key",
        "hanzi_metadata",
        "--metadata_types",
        "radical",
        "pinyin",
        "components",
    ]
    subprocess.run(cmd, check=True)


def download_opus(split: str, limit: int, hf_token: str | None) -> List[Dict]:
    ds = load_dataset("Helsinki-NLP/opus-100", "en-zh", split=split, token=hf_token)
    if limit:
        ds = ds.select(range(limit))

    prepared: List[Dict[str, str]] = []
    for row in ds:
        translation = row.get("translation") or {}
        zh = translation.get("zh", "")
        en = translation.get("en", "")
        prepared.append({"zh": zh, "en": en})
    return prepared


def annotate_zh_entries(entries: List[Dict]) -> List[Dict]:
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir) / "opus_zh.jsonl"
        annotated_path = Path(tmpdir) / "opus_zh_annotated.json"
        _write_jsonl(entries, raw_path)
        _run_annotation(raw_path, annotated_path)

        with annotated_path.open("r", encoding="utf-8") as f:
            annotated_entries = json.load(f)

    enriched: List[Dict] = []
    for entry in annotated_entries:
        zh = entry.get("zh", "")
        meta = entry.get("hanzi_metadata", {})
        enriched.append(
            {
                "zh": zh,
                "zh_pin": _build_pinyin(zh, meta),
                "zh_rad1": _build_radicals(zh, meta, "radical"),
                "zh_rad2": _build_radicals(zh, meta, "graphical"),
                "en": entry.get("en", ""),
                "hanzi_metadata": meta,
            }
        )
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OPUS-100 en-zh and add Hanzi metadata fields.")
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to download (train/validation/test).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of rows to process (use 0 for all).",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("data/annotated_zh/annotated_zh_en.json"),
        help="Where to write the annotated JSON array.",
    )
    parser.add_argument(
        "--hf_token",
        default=os.getenv("HF_TOKEN"),
        help="Optional Hugging Face token for gated repositories.",
    )
    args = parser.parse_args()

    entries = download_opus(args.split, args.limit, args.hf_token)
    annotated = annotate_zh_entries(entries)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(annotated, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {len(annotated)} annotated rows to {args.output_json}")


if __name__ == "__main__":
    main()
