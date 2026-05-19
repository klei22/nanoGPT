"""Upload a tokenizer built from a local Hugging Face tokenizer.json file.

Example:
python huggingface_model/upload_tokenizer_from_hf_json.py \
  --tokenizer_json path/to/tokenizer.json \
  --repo_id your-hf-user/your-tokenizer-repo
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast


def build_tokenizer(tokenizer_json: str, pad_token: str | None, bos_token: str | None, eos_token: str | None, unk_token: str | None) -> PreTrainedTokenizerFast:
    tk = Tokenizer.from_file(tokenizer_json)
    return PreTrainedTokenizerFast(
        tokenizer_object=tk,
        pad_token=pad_token,
        bos_token=bos_token,
        eos_token=eos_token,
        unk_token=unk_token,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload tokenizer.json to Hugging Face Hub")
    parser.add_argument("--tokenizer_json", type=str, required=True, help="Path to local tokenizer.json")
    parser.add_argument("--repo_id", type=str, required=True, help="HF repo id, e.g. user/my-tokenizer")
    parser.add_argument("--private", action="store_true", help="Create private repo")
    parser.add_argument("--pad_token", type=str, default="<|pad|>")
    parser.add_argument("--bos_token", type=str, default="<|bos|>")
    parser.add_argument("--eos_token", type=str, default="<|eos|>")
    parser.add_argument("--unk_token", type=str, default="<|unk|>")
    args = parser.parse_args()

    tokenizer_path = Path(args.tokenizer_json)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer json not found: {tokenizer_path}")

    tokenizer = build_tokenizer(
        tokenizer_json=str(tokenizer_path),
        pad_token=args.pad_token,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
        unk_token=args.unk_token,
    )

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    tokenizer.push_to_hub(args.repo_id)
    print(f"Uploaded tokenizer to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
