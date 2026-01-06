"""Zeroth-order (MeZO) training script.

This implements the MeZO update described in the paper:
  Fine-Tuning Language Models with Just Forward Passes

It supports training from scratch or resuming from a checkpoint that was
previously saved by this script (or train.py, as long as model_args match).
"""

from __future__ import annotations

import json
import os
import pickle
import time
from contextlib import nullcontext

import numpy as np
import torch

from model import GPT
from gpt_conf import GPTConfig
from train_args import parse_args
from train_variations.loss_variants import build_loss_function


def get_vocab_size_from_meta(dataset: str) -> int:
    meta_path = os.path.join("data", dataset, "meta.pkl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found at {meta_path}")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    vocab_size = meta.get("vocab_size", None)
    if vocab_size is None:
        raise ValueError("Meta file does not contain vocab_size")
    return vocab_size


def load_dataset(dataset: str, vocab_size: int):
    dtype = np.uint32 if vocab_size == 100277 else np.uint16
    train_data = np.memmap(os.path.join("data", dataset, "train.bin"), dtype=dtype, mode="r")
    val_data = np.memmap(os.path.join("data", dataset, "val.bin"), dtype=dtype, mode="r")
    return train_data, val_data


def get_batch(data, block_size: int, batch_size: int, device: torch.device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


def estimate_loss(model, data, loss_fn, ctx, eval_iters: int, block_size: int, batch_size: int, device: torch.device):
    model.eval()
    losses = torch.zeros(eval_iters)
    with torch.no_grad():
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            with ctx:
                _, loss = model(x, y, iter_num=None, loss_fn=loss_fn)
            losses[k] = loss.item()
    model.train()
    return losses.mean().item(), losses.std().item()


def build_model(args, model_group, training_group, device: torch.device):
    model_args = {action.dest: getattr(args, action.dest) for action in model_group._group_actions}
    model_args["vocab_size"] = None
    model_args["eval_interval"] = args.eval_interval

    training_args = {action.dest: getattr(args, action.dest) for action in training_group._group_actions}

    if args.init_from == "scratch":
        model_args["vocab_size"] = get_vocab_size_from_meta(args.dataset)
        config_json = {**model_args, **training_args}
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "full_config.json"), "w") as configuration_file:
            json.dump(config_json, configuration_file, indent=4)

        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        iter_num = 0
        best_val_loss = float("inf")
        best_iter = 0
    elif args.init_from in {"resume", "prev_run"}:
        if args.init_from == "resume":
            ckpt_path = os.path.join(args.out_dir, args.init_from_ckpt)
            iter_num = None
        else:
            ckpt_path = os.path.join(args.prev_run_ckpt, args.init_from_ckpt)
            iter_num = 0

        checkpoint = torch.load(ckpt_path, map_location=device)
        model_args = checkpoint["model_args"]

        altered_model_args = {
            action.dest: getattr(args, action.dest)
            for action in model_group._group_actions
            if action.default != getattr(args, action.dest)
        }
        for k in altered_model_args:
            model_args[k] = altered_model_args[k]

        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        for k, v in list(state_dict.items()):
            if k.startswith("_orig_mod."):
                state_dict[k[len("_orig_mod."):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        best_iter = checkpoint.get("best_iter", 0)
        if iter_num is None:
            iter_num = checkpoint.get("iter_num", 0)
    else:
        raise ValueError("MeZO training currently supports init_from scratch|resume|prev_run only.")

    model.to(device)
    return model, model_args, training_args, iter_num, best_val_loss, best_iter


def perturb_parameters(model, epsilon: float, seed: int, generator: torch.Generator):
    generator.manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            if not param.requires_grad:
                continue
            noise = torch.randn_like(param, generator=generator)
            param.add_(epsilon * noise)


def mezo_update(model, loss_fn, ctx, epsilon: float, learning_rate: float, seed: int, generator: torch.Generator, x, y, iter_num: int):
    perturb_parameters(model, epsilon, seed, generator)
    with torch.no_grad():
        with ctx:
            _, loss_plus = model(x, y, iter_num=iter_num, loss_fn=loss_fn)

    perturb_parameters(model, -2.0 * epsilon, seed, generator)
    with torch.no_grad():
        with ctx:
            _, loss_minus = model(x, y, iter_num=iter_num, loss_fn=loss_fn)

    perturb_parameters(model, epsilon, seed, generator)
    projected_grad = (loss_plus - loss_minus) / (2.0 * epsilon)

    generator.manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            if not param.requires_grad:
                continue
            noise = torch.randn_like(param, generator=generator)
            param.add_(-learning_rate * projected_grad * noise)

    return loss_plus.item(), loss_minus.item(), projected_grad.item()


def save_checkpoint(args, model, model_args, training_args, iter_num, best_val_loss, best_iter):
    if args.never_save_checkpoint:
        return
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": None,
        "scheduler": None,
        "model_args": model_args,
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "best_iter": best_iter,
        "best_tokens": 0,
        "config": {**training_args, **model_args},
    }
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(checkpoint, os.path.join(args.out_dir, "ckpt.pt"))


def main():
    args, model_group, training_group, _logging_group = parse_args()

    if args.training_mode != "single":
        raise ValueError("MeZO training currently supports training_mode=single only.")

    device = torch.device(args.device)
    device_type = "cuda" if "cuda" in args.device else "cpu"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    torch.manual_seed(args.seed)
    if device_type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model, model_args, training_args, iter_num, best_val_loss, best_iter = build_model(
        args, model_group, training_group, device
    )

    vocab_size = model_args["vocab_size"]
    train_data, val_data = load_dataset(args.dataset, vocab_size)

    loss_fn = build_loss_function(args)
    if hasattr(loss_fn, "set_model"):
        loss_fn.set_model(model)

    generator = torch.Generator(device=device)
    model.train()
    start_time = time.time()

    while iter_num < args.max_iters:
        x, y = get_batch(train_data, args.block_size, args.batch_size, device)
        step_seed = int(torch.randint(0, 2**31 - 1, (1,)).item()) if args.mezo_seed is None else args.mezo_seed + iter_num

        loss_plus, loss_minus, projected_grad = mezo_update(
            model=model,
            loss_fn=loss_fn,
            ctx=ctx,
            epsilon=args.mezo_epsilon,
            learning_rate=args.learning_rate,
            seed=step_seed,
            generator=generator,
            x=x,
            y=y,
            iter_num=iter_num,
        )

        if iter_num % args.log_interval == 0:
            elapsed = time.time() - start_time
            print(
                f"iter {iter_num}: loss+ {loss_plus:.4f}, loss- {loss_minus:.4f}, "
                f"proj_grad {projected_grad:.4f}, time {elapsed:.2f}s"
            )

        if iter_num % args.eval_interval == 0:
            train_loss, train_std = estimate_loss(
                model, train_data, loss_fn, ctx, args.eval_iters, args.block_size, args.batch_size, device
            )
            val_loss, val_std = estimate_loss(
                model, val_data, loss_fn, ctx, args.eval_iters, args.block_size, args.batch_size, device
            )
            print(
                f"eval iter {iter_num}: train {train_loss:.4f}±{train_std:.4f}, "
                f"val {val_loss:.4f}±{val_std:.4f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_iter = iter_num
                if args.always_save_checkpoint:
                    save_checkpoint(args, model, model_args, training_args, iter_num, best_val_loss, best_iter)

        iter_num += 1

    if not args.never_save_checkpoint and not args.only_save_checkpoint_at_end:
        save_checkpoint(args, model, model_args, training_args, iter_num, best_val_loss, best_iter)
    if args.only_save_checkpoint_at_end:
        save_checkpoint(args, model, model_args, training_args, iter_num, best_val_loss, best_iter)


if __name__ == "__main__":
    main()
