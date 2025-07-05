import os
import argparse
import subprocess
from pathlib import Path
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate and process periodic function datasets for training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data generation parameters
    gen_group = parser.add_argument_group('Data Generation')
    gen_group.add_argument('--points-per-cycle', type=int, default=100,
                        help='Number of points to sample per cycle')
    gen_group.add_argument('--num-cycles', type=int, default=10,
                        help='Number of cycles to generate for each function')
    gen_group.add_argument('--num-functions', type=int, default=4,
                        help='Number of different periodic functions to generate')
    
    # Dataset parameters
    data_group = parser.add_argument_group('Dataset Configuration')
    data_group.add_argument('--train-split', type=float, default=0.9,
                        help='Fraction of data to use for training (0.0-1.0)')
    data_group.add_argument('--dataset-prefix', type=str, default='periodic_dataset',
                        help='Prefix for output dataset directories')
    
    # Visualization parameters
    viz_group = parser.add_argument_group('Visualization')
    viz_group.add_argument('--plot-dir', type=str, default='plots',
                        help='Directory to save visualization plots')
    viz_group.add_argument('--plot-width', type=int, default=12,
                        help='Width of the plots in inches')
    viz_group.add_argument('--plot-height', type=int, default=6,
                        help='Height of the plots in inches')
    
    args = parser.parse_args()
    
    # Calculate total points
    args.total_points = args.points_per_cycle * args.num_cycles
    
    # Validate arguments
    if not 0 < args.train_split < 1:
        parser.error("train-split must be between 0 and 1")
    if args.points_per_cycle < 10:
        parser.error("points-per-cycle must be at least 10")
    if args.num_cycles < 1:
        parser.error("num-cycles must be at least 1")
    if args.num_functions < 1:
        parser.error("num-functions must be at least 1")
    
    return args

def run_step(step_num, description, cmd, check=True):
    """Run a command with proper formatting and error handling."""
    print(f"\nStep {step_num}: {description}")
    print("-" * 60)
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout.strip())
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error in step {step_num}:")
        print(f"Command failed with exit code {e.returncode}")
        if e.stdout:
            print("Output:", e.stdout.strip())
        if e.stderr:
            print("Error:", e.stderr.strip())
        return False

def verify_datasets(args):
    """Print verification information about the generated datasets."""
    print("\nDataset Verification:")
    print("-" * 60)
    print(f"Configuration:")
    print(f"  - Points per cycle: {args.points_per_cycle}")
    print(f"  - Number of cycles: {args.num_cycles}")
    print(f"  - Total points: {args.total_points:,}")
    print(f"  - Number of functions: {args.num_functions}")
    print(f"  - Train split: {args.train_split:.1%}")
    
    total_train_bytes = 0
    total_val_bytes = 0
    
    for i in range(args.num_functions):
        dataset_dir = f"{args.dataset_prefix}_{i}"
        if os.path.exists(dataset_dir):
            train_path = os.path.join(dataset_dir, "train.bin")
            val_path = os.path.join(dataset_dir, "val.bin")
            meta_path = os.path.join(dataset_dir, "meta.pkl")
            
            if all(os.path.exists(p) for p in [train_path, val_path, meta_path]):
                train_size = os.path.getsize(train_path)
                val_size = os.path.getsize(val_path)
                meta_size = os.path.getsize(meta_path)
                
                total_train_bytes += train_size
                total_val_bytes += val_size
                
                print(f"\nDataset {i}:")
                print(f"  - train.bin: {train_size:,} bytes")
                print(f"  - val.bin: {val_size:,} bytes")
                print(f"  - meta.pkl: {meta_size:,} bytes")
            else:
                print(f"\nWarning: Missing files in dataset {i}")
    
    print("\nTotal Statistics:")
    print(f"  - Total training data: {total_train_bytes:,} bytes")
    print(f"  - Total validation data: {total_val_bytes:,} bytes")
    print(f"  - Average train/val ratio: {total_train_bytes/(total_train_bytes + total_val_bytes):.2%}")

def main():
    args = parse_args()
    
    print("\nPeriodic Function Dataset Generator")
    print("=" * 60)
    print(f"Generating {args.num_functions} functions with {args.num_cycles} cycles each")
    print(f"Each cycle will have {args.points_per_cycle} points")
    print(f"Total points per function: {args.total_points:,}")
    
    # Ensure the plot directory exists
    os.makedirs(args.plot_dir, exist_ok=True)
    
    # Step 1: Generate periodic function data
    gen_cmd = [
        "python3", "generate_periodic.py",
        "--points-per-cycle", str(args.points_per_cycle),
        "--num-cycles", str(args.num_cycles),
        "--num-functions", str(args.num_functions),
        "--output", "periodic_functions.csv"
    ]
    if not run_step(1, "Generating periodic functions...", gen_cmd):
        sys.exit(1)
    
    # Step 2: Tokenize using float_csv method
    prep_cmd = [
        "python3", "prepare.py",
        "--method", "float_csv",
        "--csv_file", "periodic_functions.csv",
        "--csv_prefix", args.dataset_prefix,
        "--csv_percentage_train", str(args.train_split)
    ]
    if not run_step(2, "Tokenizing with float_csv method...", prep_cmd):
        sys.exit(1)
    
    # Step 3: Verify the datasets
    verify_datasets(args)
    
    # Step 4: Generate visualizations
    viz_cmd = [
        "python3", "verify_periodic_data.py",
        "--input-prefix", args.dataset_prefix,
        "--num-datasets", str(args.num_functions),
        "--output-dir", args.plot_dir,
        "--plot-width", str(args.plot_width),
        "--plot-height", str(args.plot_height)
    ]
    if not run_step(4, "Generating visualizations...", viz_cmd):
        print("\nWarning: Visualization step failed, but data generation was successful")
    
    # Clean up
    # if os.path.exists("periodic_functions.csv"):
    #     os.remove("periodic_functions.csv")
    #     print("\nCleaned up: Removed periodic_functions.csv")
    
    print("\nDemo completed successfully!")
    print(f"- Data directories: {args.dataset_prefix}_*")
    print(f"- Visualizations: {os.path.abspath(args.plot_dir)}")
    print(f"- Total points per function: {args.total_points:,} ({args.points_per_cycle} points × {args.num_cycles} cycles)")

if __name__ == '__main__':
    main() 
