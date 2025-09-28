"""Utility script to export Hypersphere GPT checkpoints to the Hugging Face format."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import GenerationConfig, GPT2TokenizerFast

from .modeling_hypersphere_gpt import HypersphereGPTForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        required=True,
        help="Directory containing the trained Hypersphere GPT checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to write the Hugging Face compatible artifacts",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="gpt2",
        help="Hugging Face tokenizer identifier to bundle with the export",
    )
    parser.add_argument(
        "--use_safetensors",
        action="store_true",
        help="Export weights using the safetensors format",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the exported model and tokenizer to the Hugging Face Hub",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Hub repository name to push to (e.g. username/hypersphere-gpt)",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="Optional Hugging Face access token for pushing to the hub",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the hub repository as private when pushing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = HypersphereGPTForCausalLM.from_pretrained(args.checkpoint_dir)
    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(args.output_dir, safe_serialization=args.use_safetensors)
    tokenizer.save_pretrained(args.output_dir)

    generation_config = GenerationConfig(
        max_length=model.config.block_size,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )
    generation_config.save_pretrained(args.output_dir)

    if args.push_to_hub:
        if not args.hub_model_id:
            raise ValueError("--hub_model_id must be specified when --push_to_hub is set")
        print(f"Pushing artifacts to the Hugging Face Hub repository {args.hub_model_id}")
        model.push_to_hub(
            args.hub_model_id,
            safe_serialization=args.use_safetensors,
            private=args.private,
            token=args.hub_token,
        )
        tokenizer.push_to_hub(args.hub_model_id, token=args.hub_token)
        generation_config.push_to_hub(args.hub_model_id, token=args.hub_token)

    print(f"Export complete. Artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
