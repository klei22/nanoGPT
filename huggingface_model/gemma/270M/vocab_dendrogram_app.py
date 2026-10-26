#!/usr/bin/env python3
"""Create an interactive dendrogram for Gemma 270M vocabulary tokens.

The script clusters LM-head token vectors and writes a standalone Plotly HTML file.
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import leaves_list, linkage
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="google/gemma-3-270m")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", default="average", help="SciPy linkage method")
    parser.add_argument("--metric", default="cosine", help="SciPy distance metric")
    parser.add_argument(
        "--output-html",
        default="gemma_vocab_dendrogram.html",
        help="Path to output HTML file",
    )
    return parser.parse_args()


def _token_text(tokenizer: AutoTokenizer, token_id: int) -> str:
    text = tokenizer.convert_ids_to_tokens([token_id])[0]
    clean = text.replace("\n", "\\n").replace("\t", "\\t")
    return clean if clean else "<empty>"


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(args.device)
    model.eval()

    with np.errstate(all="ignore"):
        lm_head = model.get_output_embeddings().weight.detach().float().cpu().numpy()

    vocab_size = lm_head.shape[0]
    sample_size = min(max(args.sample_size, 2), vocab_size)
    sampled_ids = np.random.choice(vocab_size, size=sample_size, replace=False)
    sampled_vectors = lm_head[sampled_ids]

    labels = [f"{token_id}: {html.escape(_token_text(tokenizer, int(token_id)))}" for token_id in sampled_ids]

    z = linkage(sampled_vectors, method=args.method, metric=args.metric)
    order = leaves_list(z)
    ordered_ids = sampled_ids[order]

    fig = ff.create_dendrogram(
        sampled_vectors,
        labels=labels,
        linkagefun=lambda _: z,
        orientation="left",
    )
    fig.update_layout(
        title=(
            f"Gemma 270M Vocabulary Dendrogram "
            f"(n={sample_size}, method={args.method}, metric={args.metric})"
        ),
        width=1200,
        height=max(900, int(sample_size * 16)),
    )

    ordered_preview = ", ".join(str(int(x)) for x in ordered_ids[:20])
    fig.add_annotation(
        text=f"Leaf order token-id preview: [{ordered_preview}]",
        xref="paper",
        yref="paper",
        x=0,
        y=1.03,
        showarrow=False,
        align="left",
    )

    output_path = Path(args.output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")

    print(f"Saved interactive dendrogram to: {output_path}")
    print("Tip: reduce --sample-size if clustering is slow or too dense.")


if __name__ == "__main__":
    main()
