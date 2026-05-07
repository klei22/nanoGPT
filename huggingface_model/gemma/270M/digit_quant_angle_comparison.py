#!/usr/bin/env python3
"""Compare pairwise digit-token angles before/after symmetric quantization."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Digit angle comparison across quantization levels")
    p.add_argument("--model", default="google/gemma-3-270m")
    p.add_argument("--embedding-source", choices=["input", "lm_head"], default="input")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default="./gemma_digit_quant_angles")
    return p.parse_args()


def _resolve_digit_ids(tokenizer: AutoTokenizer) -> dict[str, int]:
    out: dict[str, int] = {}
    for d in range(10):
        text = str(d)
        tid = tokenizer.convert_tokens_to_ids(text)
        if tid is None or tid == tokenizer.unk_token_id:
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if not ids:
                raise ValueError(f"Could not tokenize digit {text}")
            tid = int(ids[0])
        out[text] = int(tid)
    return out


def _symmetric_quantize(vecs: np.ndarray, mode: str) -> np.ndarray:
    x = vecs.copy()
    max_abs = np.max(np.abs(x), axis=1, keepdims=True)
    max_abs[max_abs == 0] = 1.0

    if mode == "binary":
        q = np.where(x >= 0, 1.0, -1.0)
        return q * max_abs
    if mode == "ternary":
        thr = 0.5 * max_abs
        q = np.zeros_like(x)
        q[x > thr] = 1.0
        q[x < -thr] = -1.0
        return q * max_abs

    bits = int(mode)
    qmax = (2 ** (bits - 1)) - 1
    scale = max_abs / qmax
    q = np.round(x / scale)
    q = np.clip(q, -qmax, qmax)
    return q * scale


def _pairwise_angles(vecs: np.ndarray) -> np.ndarray:
    n = vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
    cos = np.clip(n @ n.T, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _write_matrix_csv(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("digit," + ",".join(labels) + "\n")
        for i, d in enumerate(labels):
            vals = ",".join(f"{matrix[i, j]:.6f}" for j in range(len(labels)))
            f.write(f"{d},{vals}\n")


def _plot_per_digit(output_dir: Path, digits: list[str], angles_by_mode: dict[str, np.ndarray]) -> None:
    modes = list(angles_by_mode.keys())
    x = np.arange(len(digits))
    for i, d in enumerate(digits):
        plt.figure(figsize=(11, 5))
        for mode in modes:
            y = angles_by_mode[mode][i]
            plt.plot(x, y, marker="o", label=mode)
        plt.xticks(x, digits)
        plt.title(f"Digit {d}: relative angles to digits 0-9")
        plt.xlabel("Other digit")
        plt.ylabel("Angle (degrees)")
        plt.grid(alpha=0.3)
        plt.legend(ncol=3, fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / f"digit_{d}_relative_angles.png", dpi=180)
        plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="eager")

    emb = model.get_input_embeddings().weight.detach() if args.embedding_source == "input" else model.lm_head.weight.detach()
    emb = emb.to(args.device, dtype=torch.float32)

    digit_ids = _resolve_digit_ids(tokenizer)
    digits = [str(i) for i in range(10)]
    id_list = [digit_ids[d] for d in digits]

    vecs = emb[id_list].cpu().numpy()
    quant_modes = ["fp32", "8", "7", "6", "5", "4", "3", "ternary", "binary"]

    angles_by_mode: dict[str, np.ndarray] = {}
    for mode in quant_modes:
        qvecs = vecs if mode == "fp32" else _symmetric_quantize(vecs, mode)
        angles = _pairwise_angles(qvecs)
        angles_by_mode[mode] = angles
        _write_matrix_csv(output_dir / f"angles_{mode}.csv", digits, angles)

    with (output_dir / "digit_token_ids.csv").open("w", encoding="utf-8") as f:
        f.write("digit,token_id,token\n")
        for d in digits:
            tid = digit_ids[d]
            tok = tokenizer.convert_ids_to_tokens(tid)
            f.write(f"{d},{tid},{tok}\n")

    _plot_per_digit(output_dir, digits, angles_by_mode)

    with (output_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"model={args.model}\n")
        f.write(f"embedding_source={args.embedding_source}\n")
        f.write(f"quant_modes={','.join(quant_modes)}\n")
        f.write("outputs=angles_*.csv,digit_*_relative_angles.png,digit_token_ids.csv\n")

    print(f"Done. Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
