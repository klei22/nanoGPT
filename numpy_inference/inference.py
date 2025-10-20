import argparse
import numpy as np

from . import Transformer, load_weights


def main():
    parser = argparse.ArgumentParser(description="Numpy inference from exported weights")
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to weights file produced by save_weights.py')
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--vocab_size', type=int, default=50304)
    parser.add_argument('--tokens', type=int, default=20, help='Number of tokens to generate')
    parser.add_argument('--quant_bits', type=int, default=None,
                        help='If set, quantize weights to this many bits')
    parser.add_argument('--top_k', type=int, default=40,
                        help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-p nucleus sampling parameter')
    args = parser.parse_args()

    weights = load_weights(args.weights)
    model = Transformer(args.n_layer, args.n_head, args.n_embd,
                        args.block_size, args.vocab_size,
                        weights, args.quant_bits)

    idx = np.zeros((1, 1), dtype=np.int32)
    out = model.generate(idx, args.tokens, top_k=args.top_k, top_p=args.top_p)
    print(out)


if __name__ == '__main__':
    main()

