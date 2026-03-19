# Grokking Dataset

Modular arithmetic datasets for studying the **grokking** phenomenon
(Power et al., 2022), where neural networks first memorize training data and
then, much later in training, suddenly generalize to the validation set.

## Quick Start

```bash
# From the nanoGPT root directory:
bash demos/grokking_demo.sh
```

## Manual Usage

### 1. Generate the dataset

```bash
cd data/grokking
python3 generate_dataset.py --prime 97 --operation addition --train_fraction 0.5 --seed 42
```

This creates `input.txt`, `train_raw.txt`, and `val_raw.txt`.

### 2. Tokenize

```bash
python3 ../template/prepare.py -t input.txt --method char
```

This creates `train.bin`, `val.bin`, and `meta.pkl`.

### 3. Train

See `demos/grokking_demo.sh` for recommended training hyperparameters.

## Supported Operations

| Operation    | Formula                   | Flag           |
|-------------|---------------------------|----------------|
| Addition    | (a + b) mod p             | `--operation addition` |
| Subtraction | (a - b) mod p             | `--operation subtraction` |
| Division    | (a * b^-1) mod p          | `--operation division` |
| Polynomial  | (a^2 + a*b) mod p         | `--operation x2y` |

## Key Parameters

- `--prime`: The prime modulus (default: 97). Larger primes = more examples.
- `--train_fraction`: Fraction used for training (default: 0.5). The classic
  grokking setup uses 50% train / 50% validation.
- `--seed`: Random seed for shuffling examples.

## Background

Grokking occurs when:
1. The model quickly memorizes the training set (train loss drops to ~0)
2. Validation loss remains high for a long period
3. Eventually, validation accuracy suddenly jumps to near-perfect

Key ingredients for grokking:
- **Small dataset** relative to model capacity
- **Weight decay** (regularization is crucial for the transition)
- **Long training** (often 10x-100x beyond memorization)
