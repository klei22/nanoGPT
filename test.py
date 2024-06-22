import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
from vizier.service import clients, pyvizier as vz
from rich import print
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn

# Factorization function with configurable A
def factorize_matrix(A, output_dir, device, num_epochs, progress, task_id):
    A = int(A)  # Ensure A is an integer
    n_rows = 50000
    n_cols = 384
    original_matrix = torch.randn(n_rows, n_cols).to(device)

    W1 = torch.randn((n_rows, A), requires_grad=True, device=device)
    W2 = torch.randn((A, n_cols), requires_grad=True, device=device)

    optimizer = optim.Adam([W1, W2], lr=1e-3)
    loss_fn = nn.MSELoss()

    print(f"Starting training with A = {A}")

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        reconstructed_matrix = torch.matmul(W1, W2)
        loss = loss_fn(reconstructed_matrix, original_matrix)
        loss.backward()
        optimizer.step()

        progress.update(task_id, advance=1, description=f"Loss: {loss.item():.4f}")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the matrices to .npy files
    np.save(f"{output_dir}/W1.npy", W1.detach().cpu().numpy())
    np.save(f"{output_dir}/W2.npy", W2.detach().cpu().numpy())

    # Print the resulting matrices
    # print("W1:", W1)
    # print("W2:", W2)

    return loss.item()

def run_experiment_with_vizier(vizier_algorithm, vizier_iterations, A_start, A_end, num_epochs, device):
    search_space = vz.SearchSpace()
    search_space.root.add_int_param(name="A", min_value=A_start, max_value=A_end)

    study_config = vz.StudyConfig(
        search_space=search_space,
        metric_information=[
            vz.MetricInformation(name="loss", goal=vz.ObjectiveMetricGoal.MINIMIZE)
        ],
    )
    study_config.algorithm = vizier_algorithm
    study_client = clients.Study.from_study_config(
        study_config, owner="owner", study_id="example_study_id"
    )

    results = []
    console = Console()

    for i in range(vizier_iterations):
        print("Vizier Iteration", i)
        suggestions = study_client.suggest(count=1)
        for suggestion in suggestions:
            params = suggestion.parameters
            A = params["A"]
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.1f}%",
                refresh_per_second=1,
            ) as progress:
                task_id = progress.add_task("Training", total=num_epochs)
                loss = factorize_matrix(A, f"output_{i}_A_{A}", device, num_epochs, progress, task_id)
            suggestion.complete(vz.Measurement(metrics={"loss": loss}))
            results.append({"A": A, "loss": loss})

        # Print the results table at the end of each iteration
        results_sorted = sorted(results, key=lambda x: x["A"])
        table = Table(title=f"Results after Vizier Iteration {i + 1}")
        table.add_column("A", justify="right", style="cyan")
        table.add_column("Loss", justify="right", style="magenta")

        for result in results_sorted:
            table.add_row(str(result["A"]), f"{result['loss']:.4f}")

        console.clear()
        console.print(table)

    # Print the final sorted results
    results_sorted = sorted(results, key=lambda x: x["A"])
    table = Table(title="Final Results Sorted by A Values")
    table.add_column("A", justify="right", style="cyan")
    table.add_column("Loss", justify="right", style="magenta")

    for result in results_sorted:
        table.add_row(str(result["A"]), f"{result['loss']:.4f}")

    console.clear()
    console.print(table)

def main():
    parser = argparse.ArgumentParser(description="Run matrix factorization with Vizier optimization.")
    parser.add_argument('--vizier_algorithm', type=str, choices=[
        "GP_UCB_PE", "GAUSSIAN_PROCESS_BANDIT", "RANDOM_SEARCH", "QUASI_RANDOM_SEARCH",
        "GRID_SEARCH", "SHUFFLED_GRID_SEARCH", "EAGLE_STRATEGY", "CMA_ES",
        "EMUKIT_GP_EI", "NSGA2", "BOCS", "HARMONICA"
    ], default="GAUSSIAN_PROCESS_BANDIT", help="Choose the Vizier algorithm to use.")
    parser.add_argument('--vizier_iterations', type=int, default=20, help="Number of Vizier iterations.")
    parser.add_argument('--num_epochs', type=int, default=1000, help="Number of training epochs.")
    parser.add_argument('--A_start', type=int, default=10, help="Minimum value of A for optimization.")
    parser.add_argument('--A_end', type=int, default=100, help="Maximum value of A for optimization.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_experiment_with_vizier(
        args.vizier_algorithm,
        args.vizier_iterations,
        args.A_start,
        args.A_end,
        args.num_epochs,
        device
    )

if __name__ == "__main__":
    main()

