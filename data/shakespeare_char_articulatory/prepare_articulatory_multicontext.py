#!/usr/bin/env python3
"""Create Tiny Shakespeare multicontext lanes for case and articulatory features."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from urllib.request import urlretrieve

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
LANES = [
    "lowercase", "case", "phonetic_class", "place", "manner", "voicing",
    "vowel_height", "vowel_backness", "vowel_rounding",
]
APPROX = {
    "a": ("vowel", "open", "front", "unrounded"),
    "e": ("vowel", "mid", "front", "unrounded"),
    "i": ("vowel", "close", "front", "unrounded"),
    "o": ("vowel", "mid", "back", "rounded"),
    "u": ("vowel", "close", "back", "rounded"),
}
CONSONANTS = {
    "b": ("bilabial", "stop", "voiced"), "c": ("velar", "stop", "voiceless"),
    "d": ("alveolar", "stop", "voiced"), "f": ("labiodental", "fricative", "voiceless"),
    "g": ("velar", "stop", "voiced"), "h": ("glottal", "fricative", "voiceless"),
    "j": ("postalveolar", "affricate", "voiced"), "k": ("velar", "stop", "voiceless"),
    "l": ("alveolar", "lateral", "voiced"), "m": ("bilabial", "nasal", "voiced"),
    "n": ("alveolar", "nasal", "voiced"), "p": ("bilabial", "stop", "voiceless"),
    "q": ("velar", "stop", "voiceless"), "r": ("alveolar", "approximant", "voiced"),
    "s": ("alveolar", "fricative", "voiceless"), "t": ("alveolar", "stop", "voiceless"),
    "v": ("labiodental", "fricative", "voiced"), "w": ("labiovelar", "approximant", "voiced"),
    "x": ("velar", "fricative", "voiceless"), "y": ("palatal", "approximant", "voiced"),
    "z": ("alveolar", "fricative", "voiced"),
}
SYMBOLS = {
    "case": {"lower": "l", "upper": "u", "other": "_"},
    "phonetic_class": {"vowel": "V", "consonant": "C", "other": "_"},
    "place": {"bilabial":"B","labiodental":"F","alveolar":"A","postalveolar":"P","palatal":"Y","velar":"K","glottal":"H","labiovelar":"W","vowel":"V","other":"_"},
    "manner": {"stop":"S","fricative":"F","affricate":"A","nasal":"N","lateral":"L","approximant":"R","vowel":"V","other":"_"},
    "voicing": {"voiced":"v","voiceless":"x","vowel":"V","other":"_"},
    "vowel_height": {"close":"c","mid":"m","open":"o","consonant":"C","other":"_"},
    "vowel_backness": {"front":"f","back":"b","consonant":"C","other":"_"},
    "vowel_rounding": {"rounded":"r","unrounded":"u","consonant":"C","other":"_"},
}


def features(ch: str) -> dict[str, str]:
    lo = ch.lower()
    out = {"lowercase": lo if ch.isalpha() else ch, "case": "upper" if ch.isupper() else "lower" if ch.islower() else "other"}
    if lo in APPROX:
        _, height, backness, rounding = APPROX[lo]
        out.update(phonetic_class="vowel", place="vowel", manner="vowel", voicing="vowel", vowel_height=height, vowel_backness=backness, vowel_rounding=rounding)
    elif lo in CONSONANTS:
        place, manner, voicing = CONSONANTS[lo]
        out.update(phonetic_class="consonant", place=place, manner=manner, voicing=voicing, vowel_height="consonant", vowel_backness="consonant", vowel_rounding="consonant")
    else:
        out.update(phonetic_class="other", place="other", manner="other", voicing="other", vowel_height="other", vowel_backness="other", vowel_rounding="other")
    return out


def write_lane(root: Path, lane: str, text: str, prepare_py: Path, percentage_train: float) -> str:
    d = root / lane
    d.mkdir(parents=True, exist_ok=True)
    vals = []
    for ch in text:
        f = features(ch)
        vals.append(f["lowercase"] if lane == "lowercase" else SYMBOLS[lane][f[lane]])
    lane_text = "".join(vals)
    (d / "input.txt").write_text(lane_text, encoding="utf-8")
    if lane == "lowercase":
        token_text = "".join(sorted(set(lane_text)))
    else:
        token_text = "".join(dict.fromkeys(SYMBOLS[lane].values()))
    (d / "tokensfile.txt").write_text(token_text, encoding="utf-8")
    for link_name, target in {"prepare.py": prepare_py, "nanogpt_tokenizers.py": prepare_py.with_name("nanogpt_tokenizers.py"), "utils": prepare_py.with_name("utils")}.items():
        link_path = d / link_name
        if link_path.exists() or link_path.is_symlink():
            if link_path.is_dir() and not link_path.is_symlink():
                shutil.rmtree(link_path)
            else:
                link_path.unlink()
        link_path.symlink_to(Path(os.path.relpath(target, d)), target.is_dir())
    subprocess.run(["python3", "prepare.py", "--method", "char", "-t", "tokensfile.txt", "--percentage_train", "1.0"], cwd=d, check=True)
    subprocess.run(["python3", "prepare.py", "--method", "char", "-t", "input.txt", "--reuse_chars", "--percentage_train", str(percentage_train)], cwd=d, check=True)
    return f"{root.name}/{lane}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=None, help="Optional source text; downloads Tiny Shakespeare if omitted.")
    p.add_argument("--output_root", default="shakespeare_char_articulatory")
    p.add_argument("--percentage_train", type=float, default=0.9)
    args = p.parse_args()
    here = Path(__file__).resolve().parent
    data_root = here.parent
    root = data_root / args.output_root
    root.mkdir(parents=True, exist_ok=True)
    source = root / "source.txt"
    if args.source:
        shutil.copyfile(args.source, source)
    elif not source.exists():
        urlretrieve(URL, source)
    text = source.read_text(encoding="utf-8")
    prepare_py = data_root / "template" / "prepare.py"
    datasets = [write_lane(root, lane, text, prepare_py, args.percentage_train) for lane in LANES]
    manifest = {"source": str(source), "output_root": args.output_root, "multicontext_datasets": datasets, "lanes": LANES}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(" ".join(datasets))

if __name__ == "__main__":
    main()
