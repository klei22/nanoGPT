import argparse
import json
import os
import pickle
import time
from contextlib import nullcontext
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
from rich import print
from model import GPT, GPTConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Inference from trained models")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference")
    parser.add_argument("--out_dir", type=str, default="out", help="Directory to load checkpoint from")
    parser.add_argument("--multicontext", action=argparse.BooleanOptionalAction, help="Enable multicontext mode")
    parser.add_argument("--multicontext_datasets", type=str, default=None, help="Comma-separated list of datasets (must match wte ordering)")
    parser.add_argument("--start", type=str, default="\n", help="Start text for generation. Use '|' to separate contexts")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of inference streams to draw")
    parser.add_argument("--max_new_tokens", type=int, default=500, help="Tokens to generate in each sample")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=200, help="Retain only the top_k most likely tokens")
    parser.add_argument("--eval_only", action=argparse.BooleanOptionalAction, help="Enable evaluation only mode")
    parser.add_argument("--eval_iters", type=int, default=250, help="Iterations for evaluation")
    parser.add_argument("--eval_dataset", type=str, default=None, help="Dataset for evaluation")
    parser.add_argument("--sample_file", type=str, default=None, help="Output file for generated samples")
    return parser.parse_args()

def load_meta_files(datasets):
    """
    Load meta.pkl files for each dataset in multicontext mode.
    Returns a list of (stoi, itos, encode, decode) tuples.
    """
    meta_data = []
    for dataset in datasets:
        meta_path = os.path.join("data", dataset, "meta.pkl")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta file {meta_path} not found for dataset {dataset}")

        print(f"Loading meta from {meta_path}...")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        stoi, itos = meta["stoi"], meta["itos"]
        encode = lambda s: [stoi.get(c, stoi.get("<unk>", 0)) for c in s]
        decode = lambda l: "".join([itos[i] for i in l])

        meta_data.append((stoi, itos, encode, decode))

    return meta_data

def main():
    args = parse_args()

    if args.multicontext:
        assert args.multicontext_datasets, "You must provide --multicontext_datasets when using --multicontext"
        datasets = args.multicontext_datasets.split(",")
        meta_data = load_meta_files(datasets)  # Load meta.pkl for each dataset
    else:
        datasets, meta_data = None, None

    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type)

    checkpoint = torch.load(os.path.join(args.out_dir, "ckpt.pt"), map_location=args.device)
    model = GPT(GPTConfig(**checkpoint["model_args"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(args.device)

    if args.multicontext:
        contexts = args.start.split("|")
        assert len(contexts) == len(datasets), f"Number of contexts ({len(contexts)}) must match number of datasets ({len(datasets)})"

        token_dict, target_dict = {}, {}
        for i, (context, (_, _, encode, _)) in enumerate(zip(contexts, meta_data)):
            tokens = torch.tensor(encode(context), dtype=torch.long, device=args.device)[None, ...]
            token_dict[f"wte_{i}"] = tokens
            target_dict[f"wte_{i}"] = torch.full(tokens.shape, -1, dtype=torch.long, device=args.device)

        with torch.no_grad():
            with ctx:
                for k in range(args.num_samples):
                    output_dict = {}
                    for i, (_, _, _, decode) in enumerate(meta_data):
                        key = f"wte_{i}"
                        x = token_dict[key]
                        block_size = model.config.block_size

                        for step in range(args.max_new_tokens):
                            idx_cond = x if x.size(1) <= block_size else x[:, -block_size:]
                            logits, _ = model(idx_cond, token_dict=token_dict, target_dict=target_dict)
                            logits = logits[i]  # Select correct context's logits
                            logits = logits[:, -1, :] / args.temperature
                            if args.top_k:
                                v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
                                logits[logits < v[:, [-1]]] = -float("Inf")
                            probs = F.softmax(logits, dim=-1)
                            idx_next = torch.multinomial(probs, num_samples=1)
                            x = torch.cat((x, idx_next), dim=1)

                        output_dict[key] = decode(x[0].tolist())

                    for key, output in output_dict.items():
                        print(f"[bold green]{key}: {output}")
                    print("---------------")

                    if args.sample_file:
                        with open(args.sample_file, "w") as file:
                            file.writelines([f"{key}: {output}\n" for key, output in output_dict.items()])
    else:
        # Single context generation (for non-multicontext mode)
        encode, decode = meta_data[0][2], meta_data[0][3]
        start_ids = torch.tensor(encode(args.start), dtype=torch.long, device=args.device)[None, ...]

        with torch.no_grad():
            with ctx:
                for k in range(args.num_samples):
                    x = start_ids
                    block_size = model.config.block_size
                    for step in range(args.max_new_tokens):
                        idx_cond = x if x.size(1) <= block_size else x[:, -block_size:]
                        logits, _ = model(idx_cond)
                        logits = logits[:, -1, :] / args.temperature
                        if args.top_k:
                            v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
                            logits[logits < v[:, [-1]]] = -float("Inf")
                        probs = F.softmax(logits, dim=-1)
                        idx_next = torch.multinomial(probs, num_samples=1)
                        x = torch.cat((x, idx_next), dim=1)

                    output_text = decode(x[0].tolist())
                    print(f"[bold green]Generated: {output_text}")
                    print("---------------")

                    if args.sample_file:
                        with open(args.sample_file, "w") as file:
                            file.write(output_text + "\n")

if __name__ == "__main__":
    main()

