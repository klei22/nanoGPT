import pandas as pd
from rich.console import Console
from rich.table import Table
import argparse

def view_csv(csv_path, digits):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Initialize the console
    console = Console()
    
    # Create a table
    table = Table(title="Results from CSV")
    table.add_column("A", justify="right", style="cyan")
    table.add_column("Min Loss", justify="right", style="magenta")
    table.add_column("Max Loss", justify="right", style="magenta")
    table.add_column("Mean Loss", justify="right", style="magenta")
    table.add_column("Median Loss", justify="right", style="magenta")
    table.add_column("Std Dev", justify="right", style="magenta")

    # Add rows to the table
    for _, row in df.iterrows():
        table.add_row(
            str(row["A"]),
            f"{row['min']:.{digits}f}",
            f"{row['max']:.{digits}f}",
            f"{row['mean']:.{digits}f}",
            f"{row['median']:.{digits}f}",
            f"{row['std']:.{digits}f}"
        )

    # Print the table
    console.print(table)

def main():
    parser = argparse.ArgumentParser(description="View CSV results with rich formatting.")
    parser.add_argument('--csv_path', type=str, required=True, help="Path to the CSV file.")
    parser.add_argument('--digits', type=int, default=4, help="Number of digits to display for loss values.")
    args = parser.parse_args()
    
    view_csv(args.csv_path, args.digits)

if __name__ == "__main__":
    main()

