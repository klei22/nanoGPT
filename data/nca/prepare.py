"""
Neural Cellular Automata (NCA) data generator for nanoGPT.

Generates synthetic training data from 2D discrete neural cellular automata,
following the approach described in:
  "Training Language Models via Neural Cellular Automata" (Lee et al., 2026)

Each NCA trajectory is produced by a randomly initialized neural network
transition rule on a 12x12 grid with n states. Trajectories are tokenized
using non-overlapping 2x2 patches and serialized with grid delimiters.

Complexity filtering via gzip compression ratio retains only trajectories
with sufficient structural richness.

Usage:
  python data/nca/prepare.py [options]

  # Quick test run
  python data/nca/prepare.py --num_sequences 100 --num_timesteps 10

  # Full generation matching paper scale (~164M tokens)
  python data/nca/prepare.py --num_sequences 250000 --num_timesteps 50

  # Lower complexity band (good for code domains)
  python data/nca/prepare.py --min_gzip_ratio 30 --max_gzip_ratio 40

  # Binary alphabet (scales better with more tokens per paper)
  python data/nca/prepare.py --n_states 2
"""

import os
import argparse
import gzip
import pickle
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate NCA trajectory data for nanoGPT pre-pre-training"
    )
    # Grid and NCA parameters
    parser.add_argument("--grid_h", type=int, default=12,
                        help="Grid height")
    parser.add_argument("--grid_w", type=int, default=12,
                        help="Grid width")
    parser.add_argument("--n_states", type=int, default=10,
                        help="Number of discrete cell states (alphabet size)")
    parser.add_argument("--temperature", type=float, default=1e-3,
                        help="Softmax temperature for NCA transitions")
    parser.add_argument("--conv_channels", type=int, default=4,
                        help="Number of channels in 3x3 conv layer")
    parser.add_argument("--mlp_hidden", type=int, default=16,
                        help="Hidden size of cell-wise MLP")

    # Generation parameters
    parser.add_argument("--num_sequences", type=int, default=5000,
                        help="Number of NCA trajectories to generate")
    parser.add_argument("--num_timesteps", type=int, default=50,
                        help="Number of timesteps per trajectory")
    parser.add_argument("--max_seq_len", type=int, default=1024,
                        help="Maximum token sequence length")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Complexity filtering
    parser.add_argument("--min_gzip_ratio", type=float, default=20.0,
                        help="Minimum gzip compression ratio (%%) to retain. "
                             "Paper uses 50%% but that requires many more candidates. "
                             "20%% filters trivial fixed-point dynamics while keeping "
                             "a practical acceptance rate.")
    parser.add_argument("--max_gzip_ratio", type=float, default=100.0,
                        help="Maximum gzip compression ratio (%%) to retain")
    parser.add_argument("--max_attempts_multiplier", type=float, default=5.0,
                        help="Generate up to this many times num_sequences candidates")

    # Patch tokenization
    parser.add_argument("--patch_h", type=int, default=2,
                        help="Patch height for tokenization")
    parser.add_argument("--patch_w", type=int, default=2,
                        help="Patch width for tokenization")

    # Output
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="Fraction of sequences for validation")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (defaults to script directory)")

    return parser.parse_args()


class NCARule:
    """
    A random discrete Neural Cellular Automata transition rule.

    Architecture: 3x3 convolution (one-hot input) -> cell-wise MLP -> logits
    Following the paper: conv(n_states * 9 -> conv_channels) then
    MLP(conv_channels -> mlp_hidden -> n_states) with ReLU.
    """

    def __init__(self, n_states, conv_channels, mlp_hidden, rng):
        self.n_states = n_states
        # Conv layer: maps 3x3 neighborhood one-hot (n_states * 9) -> conv_channels
        # We implement as a gathered matmul per cell for simplicity
        input_dim = n_states * 9  # 3x3 neighborhood, each cell one-hot
        self.conv_w = rng.standard_normal((input_dim, conv_channels)).astype(np.float32) * 0.5
        self.conv_b = np.zeros(conv_channels, dtype=np.float32)

        # MLP: conv_channels -> mlp_hidden -> n_states
        self.mlp_w1 = rng.standard_normal((conv_channels, mlp_hidden)).astype(np.float32) * 0.5
        self.mlp_b1 = np.zeros(mlp_hidden, dtype=np.float32)
        self.mlp_w2 = rng.standard_normal((mlp_hidden, n_states)).astype(np.float32) * 0.5
        self.mlp_b2 = np.zeros(n_states, dtype=np.float32)

    def step(self, grid, temperature, rng):
        """
        Advance the grid by one timestep.

        Args:
            grid: (H, W) integer array with values in [0, n_states)
            temperature: softmax temperature
            rng: numpy random generator

        Returns:
            new_grid: (H, W) integer array
        """
        H, W = grid.shape
        n = self.n_states

        # One-hot encode the grid
        one_hot = np.zeros((H, W, n), dtype=np.float32)
        one_hot[np.arange(H)[:, None], np.arange(W)[None, :], grid] = 1.0

        # Pad with periodic boundaries
        padded = np.pad(one_hot, ((1, 1), (1, 1), (0, 0)), mode='wrap')

        # Extract 3x3 neighborhoods and flatten
        # For each cell (i,j), gather the 3x3 neighborhood
        neighborhoods = np.zeros((H, W, 9 * n), dtype=np.float32)
        idx = 0
        for di in range(3):
            for dj in range(3):
                neighborhoods[:, :, idx * n:(idx + 1) * n] = padded[di:di + H, dj:dj + W, :]
                idx += 1

        # Conv layer (linear transform of neighborhood)
        # (H, W, 9*n) @ (9*n, conv_channels) -> (H, W, conv_channels)
        conv_out = neighborhoods @ self.conv_w + self.conv_b

        # MLP with ReLU
        hidden = np.maximum(0, conv_out @ self.mlp_w1 + self.mlp_b1)
        logits = hidden @ self.mlp_w2 + self.mlp_b2  # (H, W, n_states)

        # Softmax with temperature
        logits_scaled = logits / temperature
        logits_scaled -= logits_scaled.max(axis=-1, keepdims=True)
        exp_logits = np.exp(logits_scaled)
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

        # Sample next state for each cell
        flat_probs = probs.reshape(-1, n)
        cumulative = np.cumsum(flat_probs, axis=-1)
        u = rng.random(flat_probs.shape[0])[:, None]
        new_states = (cumulative < u).sum(axis=-1).astype(np.int32)
        new_states = np.clip(new_states, 0, n - 1)

        return new_states.reshape(H, W)


def generate_trajectory(grid_h, grid_w, n_states, conv_channels, mlp_hidden,
                        temperature, num_timesteps, rng):
    """Generate a single NCA trajectory."""
    # Random initial grid
    grid = rng.integers(0, n_states, size=(grid_h, grid_w))

    # Random NCA rule
    rule = NCARule(n_states, conv_channels, mlp_hidden, rng)

    # Roll out trajectory
    trajectory = [grid.copy()]
    for _ in range(num_timesteps - 1):
        grid = rule.step(grid, temperature, rng)
        trajectory.append(grid.copy())

    return trajectory


def compute_gzip_ratio(trajectory):
    """Compute gzip compression ratio of a trajectory.

    Serializes the trajectory as a uint8 byte stream (one byte per cell)
    to get meaningful compression ratios, since cell values are in [0, n).
    """
    raw = np.array(trajectory, dtype=np.uint8).tobytes()
    compressed = gzip.compress(raw)
    return len(compressed) / len(raw) * 100


def tokenize_trajectory(trajectory, n_states, patch_h, patch_w,
                        grid_token, end_grid_token):
    """
    Tokenize a trajectory using 2x2 patch vocabulary.

    Each 2x2 patch of cell states maps to a unique token ID.
    Grid delimiters mark timestep boundaries.
    """
    tokens = []
    for grid in trajectory:
        H, W = grid.shape
        tokens.append(grid_token)
        # Extract non-overlapping patches in row-major order
        for i in range(0, H, patch_h):
            for j in range(0, W, patch_w):
                if i + patch_h > H or j + patch_w > W:
                    continue
                patch = grid[i:i + patch_h, j:j + patch_w]
                # Map patch to token: bijective encoding
                token_id = 0
                for pi in range(patch_h):
                    for pj in range(patch_w):
                        token_id = token_id * n_states + patch[pi, pj]
                # Offset by 2 for the two delimiter tokens
                tokens.append(token_id + 2)
        tokens.append(end_grid_token)
    return tokens


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Vocab: 2 delimiter tokens + n_states^(patch_h * patch_w) patch tokens
    num_patch_tokens = args.n_states ** (args.patch_h * args.patch_w)
    vocab_size = num_patch_tokens + 2  # +2 for <grid> and </grid>
    grid_token = 0       # <grid>
    end_grid_token = 1   # </grid>

    print(f"NCA Configuration:")
    print(f"  Grid: {args.grid_h}x{args.grid_w}, States: {args.n_states}")
    print(f"  Conv channels: {args.conv_channels}, MLP hidden: {args.mlp_hidden}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Timesteps per trajectory: {args.num_timesteps}")
    print(f"  Vocab size: {vocab_size} ({num_patch_tokens} patches + 2 delimiters)")
    print(f"  Gzip complexity band: [{args.min_gzip_ratio}%, {args.max_gzip_ratio}%]")
    print(f"  Target sequences: {args.num_sequences}")
    print()

    # Tokens per timestep: 1 (<grid>) + (H/patch_h * W/patch_w) + 1 (</grid>)
    patches_per_grid = (args.grid_h // args.patch_h) * (args.grid_w // args.patch_w)
    tokens_per_timestep = patches_per_grid + 2  # +2 for delimiters
    max_timesteps_per_seq = args.max_seq_len // tokens_per_timestep
    effective_timesteps = min(args.num_timesteps, max_timesteps_per_seq)
    print(f"  Patches per grid: {patches_per_grid}")
    print(f"  Tokens per timestep: {tokens_per_timestep}")
    print(f"  Effective timesteps per sequence (capped by max_seq_len): {effective_timesteps}")
    print()

    # Generate trajectories with complexity filtering
    all_tokens = []
    accepted = 0
    attempted = 0
    max_attempts = int(args.num_sequences * args.max_attempts_multiplier)

    while accepted < args.num_sequences and attempted < max_attempts:
        attempted += 1

        trajectory = generate_trajectory(
            args.grid_h, args.grid_w, args.n_states,
            args.conv_channels, args.mlp_hidden,
            args.temperature, effective_timesteps, rng
        )

        # Complexity filtering
        ratio = compute_gzip_ratio(trajectory)
        if ratio < args.min_gzip_ratio or ratio > args.max_gzip_ratio:
            continue

        tokens = tokenize_trajectory(
            trajectory, args.n_states,
            args.patch_h, args.patch_w,
            grid_token, end_grid_token
        )

        # Truncate to max sequence length
        tokens = tokens[:args.max_seq_len]

        all_tokens.extend(tokens)
        accepted += 1

        if accepted % max(1, args.num_sequences // 10) == 0:
            print(f"  Generated {accepted}/{args.num_sequences} sequences "
                  f"({attempted} attempted, {attempted - accepted} rejected)")

    print(f"\nGeneration complete: {accepted} sequences from {attempted} attempts")
    print(f"  Acceptance rate: {accepted / max(1, attempted) * 100:.1f}%")
    print(f"  Total tokens: {len(all_tokens):,}")

    if accepted == 0:
        print("ERROR: No sequences passed the complexity filter. "
              "Try adjusting --min_gzip_ratio or --max_gzip_ratio.")
        return

    # Split into train/val
    all_tokens = np.array(all_tokens, dtype=np.uint16)

    n_val = int(len(all_tokens) * args.val_fraction)
    n_train = len(all_tokens) - n_val
    train_ids = all_tokens[:n_train]
    val_ids = all_tokens[n_train:]

    print(f"\n  Train tokens: {len(train_ids):,}")
    print(f"  Val tokens:   {len(val_ids):,}")

    # Save binary files
    train_ids.tofile(os.path.join(output_dir, 'train.bin'))
    val_ids.tofile(os.path.join(output_dir, 'val.bin'))

    # Build stoi/itos for decode
    stoi = {'<grid>': grid_token, '</grid>': end_grid_token}
    itos = {grid_token: '<grid>', end_grid_token: '</grid>'}
    for t in range(num_patch_tokens):
        # Decode patch token back to cell values for display
        cells = []
        val = t
        for _ in range(args.patch_h * args.patch_w):
            cells.append(val % args.n_states)
            val //= args.n_states
        cells.reverse()
        label = ''.join(str(c) for c in cells)
        token_id = t + 2
        stoi[label] = token_id
        itos[token_id] = label

    # Save metadata
    meta = {
        'vocab_size': vocab_size,
        'tokenizer': 'custom',
        'stoi': stoi,
        'itos': itos,
        # NCA-specific metadata
        'nca_grid_h': args.grid_h,
        'nca_grid_w': args.grid_w,
        'nca_n_states': args.n_states,
        'nca_patch_h': args.patch_h,
        'nca_patch_w': args.patch_w,
        'nca_temperature': args.temperature,
        'nca_num_timesteps': effective_timesteps,
        'nca_min_gzip_ratio': args.min_gzip_ratio,
        'nca_max_gzip_ratio': args.max_gzip_ratio,
        'nca_num_sequences': accepted,
    }
    with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print(f"\nSaved to {output_dir}:")
    print(f"  train.bin ({os.path.getsize(os.path.join(output_dir, 'train.bin')) / 1024:.1f} KB)")
    print(f"  val.bin   ({os.path.getsize(os.path.join(output_dir, 'val.bin')) / 1024:.1f} KB)")
    print(f"  meta.pkl")


if __name__ == '__main__':
    main()
