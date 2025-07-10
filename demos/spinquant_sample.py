import argparse
import os
import pickle
import sys
from pathlib import Path

import torch
import tiktoken

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from gpt_conf import GPTConfig
from model import GPT


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    config = GPTConfig(**ckpt["model_args"])
    model = GPT(config)
    sd = ckpt["model"]
    prefix = "_orig_mod."
    for k in list(sd.keys()):
        if k.startswith(prefix):
            sd[k[len(prefix) :]] = sd.pop(k)
    model.load_state_dict(sd)
    return model, ckpt


def main():
    p = argparse.ArgumentParser(description="Generate text from a SpinQuant checkpoint")
    p.add_argument("--ckpt", default=os.path.join("spinquant_out", "ckpt_spinquant.pt"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--prompt", default="Hello")
    p.add_argument("--max_new_tokens", type=int, default=50)
    args = p.parse_args()

    model, ckpt = load_checkpoint(args.ckpt, device=args.device)
    model.to(args.device)
    model.eval()

    enc = None
    encode = None
    decode = None

    if "config" in ckpt and "dataset" in ckpt["config"]:
        meta_paths = [
            os.path.join(os.path.dirname(args.ckpt), "meta.pkl"),
            os.path.join("data", ckpt["config"]["dataset"], "meta.pkl"),
            str(REPO_ROOT / "data" / ckpt["config"]["dataset"] / "meta.pkl"),
        ]
        for mp in meta_paths:
            if os.path.exists(mp):
                with open(mp, "rb") as f:
                    meta = pickle.load(f)
                if meta.get("tokenizer") == "tiktoken":
                    enc = tiktoken.get_encoding(meta["tiktoken_encoding"])
                    encode = lambda s: enc.encode(s, allowed_special={""})
                    decode = lambda l: enc.decode(l)
                elif meta.get("tokenizer") == "sentencepiece":
                    stoi, itos = meta["stoi"], meta["itos"]
                    encode = lambda s: [stoi[c] for c in s]
                    decode = lambda l: "".join([itos[i] for i in l])
                break

    if enc is None:
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={""})
        decode = lambda l: enc.decode(l)

    idx = torch.tensor(encode(args.prompt), dtype=torch.long, device=args.device)[None, :]
    vocab = model.config.vocab_size
    if idx.max().item() >= vocab:
        raise ValueError(
            "Encoded token id exceeds vocabulary size; make sure meta.pkl was copied "
            "alongside the checkpoint or specify the correct tokenizer."
        )

    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=args.max_new_tokens)
    print(decode(out[0].tolist()))


if __name__ == "__main__":
    main()
