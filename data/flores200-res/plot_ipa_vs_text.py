#!/usr/bin/env python3
"""
plot_ipa_vs_text.py

Graphs IPA vs raw text sizes for paired files across directories:

  text/<text_*.txt>
  ipa/<ipa_text_*.txt>

Defaults:
  --text-dir text/
  --ipa-dir  ipa/

Produces:
- scatter: IPA bytes vs raw bytes
- bar: IPA/raw ratio
- bar: delta bytes (IPA - raw)
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import math

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


@dataclass
class PairStats:
    lang: str
    raw_path: Path
    ipa_path: Path
    raw_bytes: int
    ipa_bytes: int
    raw_chars: int
    ipa_chars: int
    raw_lines: int
    ipa_lines: int

    @property
    def ratio_bytes(self) -> float:
        return (self.ipa_bytes / self.raw_bytes) if self.raw_bytes else float("inf")

    @property
    def delta_bytes(self) -> int:
        return self.ipa_bytes - self.raw_bytes


def read_stats(p: Path) -> Tuple[int, int, int]:
    """
    Returns (utf8_bytes, chars, lines).
    """
    data = p.read_text(encoding="utf-8", errors="replace")
    b = len(data.encode("utf-8"))
    c = len(data)
    lines = data.count("\n") + (1 if data and not data.endswith("\n") else 0)
    return b, c, lines


def discover_pairs(text_dir: Path, ipa_dir: Path) -> List[Tuple[str, Path, Path]]:
    """
    Finds pairs across directories:
      text_dir/text_<lang>.txt
      ipa_dir/ipa_text_<lang>.txt
    """
    raw_map: Dict[str, Path] = {}
    ipa_map: Dict[str, Path] = {}

    for p in text_dir.iterdir():
        if p.is_file() and p.name.startswith("text_") and p.name.endswith(".txt"):
            lang = p.name[len("text_") : -len(".txt")]
            raw_map[lang] = p

    for p in ipa_dir.iterdir():
        if p.is_file() and p.name.startswith("ipa_text_") and p.name.endswith(".txt"):
            lang = p.name[len("ipa_text_") : -len(".txt")]
            ipa_map[lang] = p

    langs = sorted(set(raw_map) & set(ipa_map))
    return [(lang, raw_map[lang], ipa_map[lang]) for lang in langs]


def make_scatter(stats: List[PairStats], outpath: Optional[Path], title: str) -> None:
    x = [s.raw_bytes for s in stats]
    y = [s.ipa_bytes for s in stats]
    labels = [s.lang for s in stats]

    plt.figure()
    plt.scatter(x, y)

    for xi, yi, lab in zip(x, y, labels):
        plt.annotate(lab, (xi, yi), textcoords="offset points", xytext=(6, 4))

    plt.xlabel("Raw text size (UTF-8 bytes)")
    plt.ylabel("IPA text size (UTF-8 bytes)")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)

    if outpath:
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()
    else:
        plt.show()


def make_bar(
    stats: List[PairStats],
    values: List[float],
    ylabel: str,
    outpath: Optional[Path],
    title: str,
) -> None:
    langs = [s.lang for s in stats]
    plt.figure(figsize=(max(8, 0.8 * len(langs)), 5))
    plt.bar(langs, values)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)

    if outpath:
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()
    else:
        plt.show()



def _mean_std(vals: List[float]) -> Tuple[float, float]:
    """
    Population mean/std (ddof=0) over vals.
    """
    if not vals:
        return 0.0, 0.0
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    return m, math.sqrt(var)


def make_back_to_back_bar(
        stats: List[PairStats],
        outpath: Optional[Path],
        title: str = "Raw vs IPA Text Size (UTF-8 bytes)",
        ) -> None:
    """
    Back-to-back horizontal bar chart:
        - Raw text on the left (negative)
      - IPA text on the right (positive)

    Adds:
        - dotted mean lines for raw and ipa
      - dotted ±1 stddev lines for raw and ipa
    """
    langs = [s.lang for s in stats]
    raw_bytes = [float(s.raw_bytes) for s in stats]
    ipa_bytes = [float(s.ipa_bytes) for s in stats]

    raw_vals = [-b for b in raw_bytes]   # negative for left side
    ipa_vals = ipa_bytes                # positive for right side

    raw_mean, raw_std = _mean_std(raw_bytes)
    ipa_mean, ipa_std = _mean_std(ipa_bytes)

    y = range(len(langs))

    plt.figure(figsize=(10, max(5, 0.5 * len(langs))))
    plt.barh(y, raw_vals, label="Raw text", alpha=0.7)
    plt.barh(y, ipa_vals, label="IPA text", alpha=0.7)

    plt.yticks(y, langs)
    plt.axvline(0, color="black", linewidth=1)

    # Mean lines (dotted)
    plt.axvline(-raw_mean, linestyle=":", linewidth=2, label=f"Raw mean ({raw_mean:.0f})")
    plt.axvline(ipa_mean, linestyle=":", linewidth=2, label=f"IPA mean ({ipa_mean:.0f})")

    # ±1 stddev lines (dotted, lighter)
    # Raw side
    plt.axvline(-(raw_mean - raw_std), linestyle=":", linewidth=1)
    plt.axvline(-(raw_mean + raw_std), linestyle=":", linewidth=1)
    # IPA side
    plt.axvline(ipa_mean - ipa_std, linestyle=":", linewidth=1)
    plt.axvline(ipa_mean + ipa_std, linestyle=":", linewidth=1)

    plt.xlabel("UTF-8 bytes")
    plt.title(
            f"{title}\n"
            f"Raw mean={raw_mean:.0f}, std={raw_std:.0f} | "
            f"IPA mean={ipa_mean:.0f}, std={ipa_std:.0f}"
            )
    plt.grid(True, axis="x", linestyle="--", linewidth=0.5)

    # Symmetric x-limits
    max_val = max(max(ipa_vals), max(abs(v) for v in raw_vals))
    # also include mean±std in bounds
    max_val = max(
            max_val,
            raw_mean + raw_std,
            ipa_mean + ipa_std,
            )
    plt.xlim(-max_val * 1.15, max_val * 1.15)

    plt.legend(loc="best")

    if outpath:
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()
    else:
        plt.show()

    # Optional: print summary to console (handy)
    print(f"[back-to-back] Raw bytes: mean={raw_mean:.2f}, std={raw_std:.2f}")
    print(f"[back-to-back] IPA bytes: mean={ipa_mean:.2f}, std={ipa_std:.2f}")

def write_csv(stats: List[PairStats], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "lang",
                "raw_path",
                "ipa_path",
                "raw_bytes",
                "ipa_bytes",
                "ratio_bytes",
                "delta_bytes",
                "raw_chars",
                "ipa_chars",
                "raw_lines",
                "ipa_lines",
            ]
        )
        for s in stats:
            w.writerow(
                [
                    s.lang,
                    str(s.raw_path),
                    str(s.ipa_path),
                    s.raw_bytes,
                    s.ipa_bytes,
                    f"{s.ratio_bytes:.6f}",
                    s.delta_bytes,
                    s.raw_chars,
                    s.ipa_chars,
                    s.raw_lines,
                    s.ipa_lines,
                ]
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Graph IPA vs raw text sizes across folders.")
    ap.add_argument("--text-dir", default="text", help="Directory with text_<lang>.txt files (default: text/)")
    ap.add_argument("--ipa-dir", default="ipa", help="Directory with ipa_text_<lang>.txt files (default: ipa/)")
    ap.add_argument("--save", action="store_true", help="Save plots instead of showing them.")
    ap.add_argument("--outdir", default="plots_out", help="Output directory when --save is used.")
    ap.add_argument("--csv", action="store_true", help="Also write CSV of statistics.")
    ap.add_argument(
        "--sort",
        choices=["lang", "raw_bytes", "ipa_bytes", "ratio", "delta"],
        default="lang",
        help="Sort order for plots.",
    )
    ap.add_argument(
        "--title",
        default="IPA vs Raw Text Size (UTF-8 bytes)",
        help="Title for scatter plot.",
    )
    args = ap.parse_args()

    text_dir = Path(args.text_dir)
    ipa_dir = Path(args.ipa_dir)

    if not text_dir.exists():
        raise SystemExit(f"text-dir not found: {text_dir}")
    if not ipa_dir.exists():
        raise SystemExit(f"ipa-dir not found: {ipa_dir}")

    pairs = discover_pairs(text_dir, ipa_dir)
    if not pairs:
        raise SystemExit("No matching text_/ipa_text_ pairs found.")

    stats: List[PairStats] = []
    for lang, raw_p, ipa_p in pairs:
        rb, rc, rl = read_stats(raw_p)
        ib, ic, il = read_stats(ipa_p)
        stats.append(
            PairStats(
                lang=lang,
                raw_path=raw_p,
                ipa_path=ipa_p,
                raw_bytes=rb,
                ipa_bytes=ib,
                raw_chars=rc,
                ipa_chars=ic,
                raw_lines=rl,
                ipa_lines=il,
            )
        )

    # Sorting
    key_map = {
        "lang": lambda s: s.lang,
        "raw_bytes": lambda s: s.raw_bytes,
        "ipa_bytes": lambda s: s.ipa_bytes,
        "ratio": lambda s: s.ratio_bytes,
        "delta": lambda s: s.delta_bytes,
    }
    stats.sort(key=key_map[args.sort])

    outdir = Path(args.outdir)
    if args.save:
        outdir.mkdir(parents=True, exist_ok=True)

    make_scatter(
        stats,
        outdir / "scatter_ipa_vs_raw_bytes.png" if args.save else None,
        args.title,
    )

    make_bar(
        stats,
        [s.ratio_bytes for s in stats],
        "IPA / Raw (bytes)",
        outdir / "bar_ratio_ipa_over_raw.png" if args.save else None,
        "IPA expansion ratio by language",
    )

    make_bar(
        stats,
        [float(s.delta_bytes) for s in stats],
        "IPA - Raw (bytes)",
        outdir / "bar_delta_ipa_minus_raw.png" if args.save else None,
        "Absolute size increase (IPA − Raw)",
    )

    make_back_to_back_bar(
            stats,
            outdir / "bar_back_to_back_raw_vs_ipa.png" if args.save else None,
            title="Raw vs IPA Text Size by Language (UTF-8 bytes)",
    )


    if args.save and args.csv:
        write_csv(stats, outdir / "ipa_vs_raw_stats.csv")

    for s in stats:
        print(
            f"{s.lang:14s} raw={s.raw_bytes:8d} "
            f"ipa={s.ipa_bytes:8d} "
            f"ratio={s.ratio_bytes:6.3f} "
            f"delta={s.delta_bytes:8d}"
        )


if __name__ == "__main__":
    main()

