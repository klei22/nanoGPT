import argparse
import os
import torch

from gpt_conf import GPTConfig
from model import GPT
from quantization.spin_quant import SpinQuant


def load_checkpoint(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    config = GPTConfig(**ckpt["model_args"])
    model = GPT(config)
    sd = ckpt["model"]
    prefix = "_orig_mod."
    for k in list(sd.keys()):
        if k.startswith(prefix):
            sd[k[len(prefix):]] = sd.pop(k)
    model.load_state_dict(sd)
    return model


def save_checkpoint(model, out_path):
    from dataclasses import asdict

    ckpt = {
        "model_args": asdict(model.config),
        "model": model.state_dict(),
    }
    torch.save(ckpt, out_path)


def main():
    p = argparse.ArgumentParser(description="Run SpinQuant PTQ on a checkpoint")
    p.add_argument("--in_dir", default="out", help="Directory with ckpt.pt from training")
    p.add_argument("--out_dir", default="spinquant_out", help="Where to write the quantized checkpoint")
    p.add_argument("--bits", type=int, default=4, help="Quantization bits")
    p.add_argument("--steps", type=int, default=100, help="Optimization steps")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--calib_batches", type=int, default=8)
    args = p.parse_args()

    ckpt_path = os.path.join(args.in_dir, "ckpt.pt")
    model = load_checkpoint(ckpt_path, device=args.device).to(args.device)

    sq = SpinQuant(model, bits=args.bits)
    vocab = model.config.vocab_size
    calib = [torch.randint(0, vocab, (1, args.seq_len), device=args.device) for _ in range(args.calib_batches)]
    sq.optimize(calib, steps=args.steps)
    sq.apply()

    os.makedirs(args.out_dir, exist_ok=True)
    out_ckpt = os.path.join(args.out_dir, "ckpt_spinquant.pt")
    save_checkpoint(model, out_ckpt)
    print(f"Saved quantized checkpoint to {out_ckpt}")


if __name__ == "__main__":
    main()
