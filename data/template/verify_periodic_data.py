import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import argparse
from pathlib import Path

def load_and_plot_dataset(dataset_idx, input_prefix="periodic_dataset", output_dir=".", plot_size=(12, 6)):
    """Load and plot both training and validation data from a dataset.

    Args:
        dataset_idx (int): Index of the dataset to process
        input_prefix (str): Prefix for the dataset directory names
        output_dir (str): Directory to save the plots
        plot_size (tuple): Size of the plot in inches (width, height)

    Returns:
        tuple: (train_length, val_length, metadata)
    """
    dataset_dir = f"{input_prefix}_{dataset_idx}"

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory {dataset_dir} not found")

    # Load the data
    train_path = os.path.join(dataset_dir, "train.bin")
    val_path = os.path.join(dataset_dir, "val.bin")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Missing data files in {dataset_dir}")

    train_data = np.fromfile(train_path, dtype=np.float16)
    val_data = np.fromfile(val_path, dtype=np.float16)

    # Load metadata
    meta_path = os.path.join(dataset_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found in {dataset_dir}")

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    # Create the plot
    plt.figure(figsize=plot_size)

    # Plot training data
    plt.plot(train_data, label='Training Data', alpha=0.7)

    # Plot validation data (offset to align with the end of training data)
    plt.plot(range(len(train_data), len(train_data) + len(val_data)),
             val_data, label='Validation Data', alpha=0.7)

    plt.title(f'Periodic Function {dataset_idx}')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    output_path = os.path.join(output_dir, f'periodic_function_{dataset_idx}.png')
    plt.savefig(output_path)
    plt.close()

    return len(train_data), len(val_data), meta

def print_metadata(meta, dataset_idx):
    """Print formatted metadata information."""
    print(f"\nDataset {dataset_idx} Metadata:")
    print("  " + "-" * 40)
    for key, value in meta.items():
        if isinstance(value, (dict, list)):
            print(f"  {key}:")
            if isinstance(value, dict):
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                for item in value:
                    print(f"    - {item}")
        else:
            print(f"  {key}: {value}")

def main():
    parser = argparse.ArgumentParser(description='Verify periodic function datasets')
    parser.add_argument('--input-prefix', type=str, default='periodic_dataset',
                      help='Prefix for input dataset directories')
    parser.add_argument('--num-datasets', type=int, default=4,
                      help='Number of datasets to verify')
    parser.add_argument('--output-dir', type=str, default='.',
                      help='Directory to save the plots')
    parser.add_argument('--plot-width', type=int, default=12,
                      help='Width of the plots in inches')
    parser.add_argument('--plot-height', type=int, default=6,
                      help='Height of the plots in inches')

    args = parser.parse_args()

    print(f"Verifying {args.num_datasets} periodic datasets...")
    print(f"Input prefix: {args.input_prefix}")
    print(f"Output directory: {args.output_dir}")

    total_train = 0
    total_val = 0
    success_count = 0

    for i in range(args.num_datasets):
        try:
            train_len, val_len, meta = load_and_plot_dataset(
                i,
                input_prefix=args.input_prefix,
                output_dir=args.output_dir,
                plot_size=(args.plot_width, args.plot_height)
            )

            print_metadata(meta, i)
            print(f"  Training samples: {train_len}")
            print(f"  Validation samples: {val_len}")

            total_train += train_len
            total_val += val_len
            success_count += 1

        except Exception as e:
            print(f"\nError processing dataset {i}: {str(e)}")

    print("\nVerification Summary:")
    print(f"Successfully processed: {success_count}/{args.num_datasets} datasets")
    print(f"Total training samples: {total_train}")
    print(f"Total validation samples: {total_val}")
    print(f"\nPlots have been saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
