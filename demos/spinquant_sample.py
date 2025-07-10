import argparse
import os
import torch
import tiktoken
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
            sd[k[len(prefix):]] = sd.pop(k)
    model.load_state_dict(sd)
    return model


def main():
    p = argparse.ArgumentParser(description="Generate text from a SpinQuant checkpoint")
    p.add_argument("--ckpt", default=os.path.join("spinquant_out", "ckpt_spinquant.pt"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--prompt", default="Hello")
    p.add_argument("--max_new_tokens", type=int, default=50)
    args = p.parse_args()

    model = load_checkpoint(args.ckpt, device=args.device).to(args.device)
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    idx = torch.tensor(enc.encode(args.prompt), dtype=torch.long, device=args.device)[None, :]
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=args.max_new_tokens)
    print(enc.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
