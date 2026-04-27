# find_gemma_vocab_groups.py
# pip install torch transformers scikit-learn sentencepiece tqdm pandas

import argparse
import json
from collections import defaultdict

import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from transformers import AutoModelForCausalLM, AutoTokenizer


def clean_token(tok: str) -> str:
    return (
        tok.replace("▁", " ")
           .replace("\n", "\\n")
           .replace("\t", "\\t")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-270m-it")
    parser.add_argument("--k", type=int, default=256, help="number of spherical-cap groups")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--top-n", type=int, default=50, help="tokens to print per group")
    parser.add_argument("--out-json", default="gemma_vocab_groups.json")
    parser.add_argument("--out-csv", default="gemma_vocab_groups.csv")
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    W = model.lm_head.weight.detach().float().cpu()  # [vocab, hidden]
    vocab_size, dim = W.shape
    print(f"lm_head shape: vocab={vocab_size}, dim={dim}")

    # Normalize rows to directions on the unit sphere
    norms = W.norm(dim=1).clamp_min(1e-8)
    X = W / norms[:, None]

    print(f"Clustering into {args.k} spherical-cap groups...")
    kmeans = MiniBatchKMeans(
        n_clusters=args.k,
        batch_size=args.batch_size,
        n_init="auto",
        random_state=0,
        verbose=0,
    )

    labels_np = kmeans.fit_predict(X.numpy())
    labels = torch.tensor(labels_np, dtype=torch.long)

    centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    centers = F.normalize(centers, dim=1)

    # Cosine similarity to assigned cap center
    assigned_centers = centers[labels]
    cos_to_center = (X * assigned_centers).sum(dim=1)
    angle_deg = torch.rad2deg(torch.acos(cos_to_center.clamp(-1, 1)))

    rows = []
    groups = defaultdict(list)

    print("Decoding tokens...")
    for token_id in range(vocab_size):
        raw_tok = tokenizer.convert_ids_to_tokens(token_id)
        text = clean_token(raw_tok)

        group_id = int(labels[token_id])
        row = {
            "token_id": token_id,
            "group": group_id,
            "token": raw_tok,
            "display_token": text,
            "norm": float(norms[token_id]),
            "cos_to_center": float(cos_to_center[token_id]),
            "angle_deg": float(angle_deg[token_id]),
        }

        rows.append(row)
        groups[group_id].append(row)

    # Sort each group by closeness to cap center
    for g in groups:
        groups[g].sort(key=lambda r: r["cos_to_center"], reverse=True)

    print("\n=== Group Summary ===")
    summary = []
    for g in range(args.k):
        items = groups[g]
        if not items:
            continue

        mean_cos = sum(x["cos_to_center"] for x in items) / len(items)
        mean_angle = sum(x["angle_deg"] for x in items) / len(items)

        top_tokens = [x["display_token"] for x in items[:args.top_n]]

        summary.append({
            "group": g,
            "size": len(items),
            "mean_cos_to_center": mean_cos,
            "mean_angle_deg": mean_angle,
            "top_tokens": top_tokens,
        })

        print(f"\nGroup {g} | size={len(items)} | mean_cos={mean_cos:.4f} | mean_angle={mean_angle:.2f}°")
        print(" ".join(repr(t) for t in top_tokens[:args.top_n]))

    print(f"\nSaving CSV to {args.out_csv}")
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)

    print(f"Saving JSON to {args.out_json}")
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "k": args.k,
                "vocab_size": vocab_size,
                "hidden_dim": dim,
                "summary": summary,
                "groups": {str(k): v for k, v in groups.items()},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Done.")


if __name__ == "__main__":
    main()
