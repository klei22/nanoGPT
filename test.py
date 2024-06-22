import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from vizier.service import clients, pyvizier as vz

# Factorization function with configurable A
def factorize_matrix(A, output_dir, device):
    A = int(A)  # Ensure A is an integer
    n_rows = 50000
    n_cols = 384
    original_matrix = torch.randn(n_rows, n_cols).to(device)

    W1 = torch.randn((n_rows, A), requires_grad=True, device=device)
    W2 = torch.randn((A, n_cols), requires_grad=True, device=device)

    optimizer = optim.Adam([W1, W2], lr=1e-3)
    loss_fn = nn.MSELoss()

    num_epochs = 1000

    print(f"Starting training with A = {A}")

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        reconstructed_matrix = torch.matmul(W1, W2)
        loss = loss_fn(reconstructed_matrix, original_matrix)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the matrices to .npy files
    np.save(f"{output_dir}/W1.npy", W1.detach().cpu().numpy())
    np.save(f"{output_dir}/W2.npy", W2.detach().cpu().numpy())

    # Print the resulting matrices
    print("W1:", W1)
    print("W2:", W2)

    return loss.item()

def run_experiment_with_vizier(vizier_algorithm, vizier_iterations, device):
    search_space = vz.SearchSpace()
    search_space.root.add_int_param(name="A", min_value=10, max_value=100)

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

    for i in range(vizier_iterations):
        print("Vizier Iteration", i)
        suggestions = study_client.suggest(count=1)
        for suggestion in suggestions:
            params = suggestion.parameters
            A = params["A"]
            loss = factorize_matrix(A, f"output_{i}_A_{A}", device)
            suggestion.complete(vz.Measurement(metrics={"loss": loss}))

    optimal_trials = study_client.optimal_trials()
    for trial in optimal_trials:
        best_trial = trial.materialize()
        print(f"Best trial: {best_trial.parameters}, Loss: {best_trial.final_measurement.metrics['loss']}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vizier_algorithm = "GAUSSIAN_PROCESS_BANDIT"
    vizier_iterations = 20
    run_experiment_with_vizier(vizier_algorithm, vizier_iterations, device)

if __name__ == "__main__":
    main()

