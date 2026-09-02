#!/usr/bin/env python3
"""Build leak-free meter/rhyme training and benchmark splits."""
import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


def normalized(text):
    return " ".join(re.findall(r"[\w']+", text.casefold()))


def content_hash(text):
    return hashlib.sha256(normalized(text).encode()).hexdigest()


def canonicalize_rhyme(labels):
    """Rename rhyme classes in order of first appearance (c d c d -> ABAB)."""
    mapping, output = {}, []
    for label in labels:
        if label in {"*", "-", "_"}:
            output.append("X")
            continue
        if label not in mapping:
            number = len(mapping)
            mapping[label] = chr(65 + number) if number < 26 else f"A{number}"
        output.append(mapping[label])
    return "".join(output)


def expand_rhyme(value, line_count):
    labels = value.split()
    if len(labels) == 3 and labels[-1] == "*" and line_count >= 2:
        labels = [labels[i % 2] for i in range(line_count)]
    elif len(labels) == 1 and len(labels[0]) == line_count:
        labels = list(labels[0])
    return labels


def split_from_path(path):
    name = path.name.casefold()
    if ".test." in name or name.startswith("test"):
        return "test"
    if ".dev." in name or "validation" in name or name.startswith("dev"):
        return "val"
    return "train"


def parse_haider(root):
    records = []
    candidates = [p for p in root.rglob("*") if p.is_file() and
                  "english" in str(p).casefold() and "meter" in p.name.casefold() and
                  p.suffix in {".txt", ".tsv"} and ".all." not in p.name]
    for path in sorted(candidates):
        rows, line_number = [], 0
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines() + [""]:
            if raw.strip():
                cols = raw.split("\t")
                if len(cols) >= 9 and cols[0].isdigit():
                    rows.append(cols)
                continue
            if not rows:
                continue
            line_number += 1
            text = "".join(row[1] if i == 0 else (row[1] if row[1].startswith(("'", "-")) else " " + row[1])
                           for i, row in enumerate(rows)).strip()
            stress = "".join("1" if row[2] == "+" else "0" for row in rows)
            coarse_index, measure_index, scansion_index = (6, 7, 8) if len(rows[0]) == 9 else (9, 10, 11)
            records.append({"id": f"haider:{path.stem}:{line_number:05d}",
                            "group_id": f"haider:{path.stem}:{line_number:05d}", "source": "haider",
                            "task": "meter", "text": text, "meter": rows[0][coarse_index],
                            "measure": rows[0][measure_index], "syllable_stress": stress,
                            "scansion": rows[0][scansion_index], "split": split_from_path(path)})
            rows = []
    return records


def parse_chicago(root):
    """Parse the Chicago AUTHOR/TITLE/RHYME/RHYME-POEM record state machine."""
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in {".txt", ".poem", ".data"}:
            continue
        author, title, lines, scheme, poem_id = path.stem, "", [], None, 0

        def emit():
            nonlocal lines, scheme, poem_id
            if not lines or not scheme:
                lines, scheme = [], None
                return
            labels = expand_rhyme(scheme, len(lines))
            if len(labels) != len(lines):
                lines, scheme = [], None
                return
            poem_id += 1
            records.append({"id": f"chicago:{path.stem}:{poem_id:05d}",
                            "group_id": f"poet:{normalized(author)}", "source": "chicago",
                            "task": "rhyme_scheme", "text": "\n".join(lines),
                            "rhyme_scheme": canonicalize_rhyme(labels), "author": author,
                            "title": title})
            lines, scheme = [], None

        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines() + ["RHYME-POEM"]:
            key, _, value = raw.partition(" ")
            if key == "AUTHOR":
                author = value.strip()
            elif key == "TITLE":
                title = value.strip()
            elif key == "RHYME":
                if lines:
                    emit()
                scheme = value.strip()
            elif key == "RHYME-POEM":
                emit()
            elif raw.strip() and scheme is not None:
                lines.append(raw.strip())
    return records


def assign_chicago_splits(records, seed):
    poets = sorted({r["group_id"] for r in records})
    random.Random(seed).shuffle(poets)
    n = len(poets)
    val, test = set(poets[: max(1, n // 10)]), set(poets[max(1, n // 10): max(2, n // 5)])
    for record in records:
        record["split"] = "val" if record["group_id"] in val else "test" if record["group_id"] in test else "train"


def serialize(record):
    if record["task"] == "meter":
        return (f"<|task|>meter\n<|poem|>{record['text']}\n<|answer|><|meter|>{record['meter']}\n"
                f"<|scansion|>{record['syllable_stress']}\n<|end|>")
    prediction = (f"<|task|>rhyme_scheme\n<|poem|>{record['text']}\n"
                  f"<|answer|><|rhyme|>{record['rhyme_scheme']}\n<|end|>")
    composition = (f"<|task|>compose\n<|rhyme|>{record['rhyme_scheme']}\n"
                   f"<|answer|><|poem|>{record['text']}\n<|end|>")
    return prediction + "\n" + composition


def benchmark_rows(record):
    row = {k: v for k, v in record.items() if k not in {"author", "title", "split"}}
    row["context"] = f"<|task|>{record['task']}\n<|poem|>{record['text']}\n<|answer|>"
    if record["task"] == "meter":
        broken_stress = record["syllable_stress"].translate(str.maketrans("01", "10"))
        correct = f"<|meter|>{record['meter']}\n<|scansion|>{record['syllable_stress']}"
        broken = f"<|meter|>{record['meter']}\n<|scansion|>{broken_stress}"
        row.update({"task": "meter_minimal_pair", "choices": [correct, broken], "label": 0,
                    "choice_meters": [record["meter"], record["meter"]],
                    "choice_scansions": [record["syllable_stress"], broken_stress]})
    else:
        correct = record["rhyme_scheme"]
        broken = correct[:-1] + ("Z" if correct[-1:] != "Z" else "Y")
        row.update({"task": "rhyme_minimal_pair", "choices": [f"<|rhyme|>{correct}", f"<|rhyme|>{broken}"],
                    "label": 0, "choice_rhymes": [correct, broken]})
    # Balance answer position deterministically without making builds seed-dependent.
    if int(record["content_hash"][-1], 16) % 2:
        row["choices"].reverse()
        for key in ("choice_meters", "choice_scansions", "choice_rhymes"):
            if key in row:
                row[key].reverse()
        row["label"] = 1
    return [row]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    records = parse_haider(args.raw_dir / "haider")
    chicago = parse_chicago(args.raw_dir / "chicago")
    assign_chicago_splits(chicago, args.seed)
    records += chicago
    accepted, seen, rejected = [], {}, Counter()
    for record in records:
        digest = content_hash(record["text"])
        if not digest or digest in seen:
            rejected["duplicate_or_empty"] += 1
            continue
        record["content_hash"] = digest
        seen[digest] = record["split"]
        accepted.append(record)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_hashes = defaultdict(set)
    for split in ("train", "val", "test"):
        selected = [r for r in accepted if r["split"] == split]
        split_hashes[split] = {r["content_hash"] for r in selected}
        rows = selected if split != "test" else [row for record in selected for row in benchmark_rows(record)]
        (args.output_dir / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        (args.output_dir / f"{split}.txt").write_text("\n".join(serialize(r) for r in selected) + ("\n" if selected else ""))
    leakage = {f"{a}_{b}": len(split_hashes[a] & split_hashes[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    train_values = {field: {r.get(field) for r in accepted if r["split"] == "train" and r.get(field)}
                    for field in ("meter", "measure", "rhyme_scheme")}
    oov = {split: {field: sum(r.get(field) is not None and r.get(field) not in train_values[field]
                               for r in accepted if r["split"] == split)
                   for field in train_values} for split in ("val", "test")}
    manifest = {"seed": args.seed, "source_counts": Counter(r["source"] for r in accepted),
                "split_counts": Counter(r["split"] for r in accepted), "rejected_records": rejected,
                "oov_counts": oov, "split_leakage": leakage,
                "checksums": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in args.output_dir.iterdir() if p.suffix in {".jsonl", ".txt"}}}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
