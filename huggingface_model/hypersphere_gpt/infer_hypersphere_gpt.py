import argparse

import torch

from .modeling_hypersphere_gpt import HypersphereGPTForCausalLM
from .tokenizer import TiktokenTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a trained Hypersphere GPT model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model directory")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = TiktokenTokenizer()

    model = HypersphereGPTForCausalLM.from_pretrained(args.model_path).to(device)
    model.eval()

    input_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    output_text = tokenizer.decode(generated[0].tolist())
    print(output_text)


if __name__ == "__main__":
    main()
