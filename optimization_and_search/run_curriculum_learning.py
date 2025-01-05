import subprocess
import argparse
import os
import json

prev_csv_dir = ""
prev_output_dir = ""

def run_experiments_command(training_stage, config, **kwargs):
    global prev_csv_dir
    global prev_output_dir

    dataset = os.path.splitext(config)[0]
    csv_dir = f"{training_stage}_{dataset}"
    output_dir = f"{training_stage}_{dataset}"
    # base command
    command = ["python3", "run_experiments.py"]
    command.extend(["--config", f"explorations/{config}"])
    # directory to output csv logs
    command.extend(["--csv_ckpt_dir", csv_dir])
    # directory to output ckpts
    command.extend(["--output_dir", output_dir])
    if prev_csv_dir and prev_output_dir:
        command.extend(["--use-best-val-loss-from", "csv_logs/" + prev_csv_dir, prev_output_dir])
    prev_csv_dir = csv_dir
    prev_output_dir = output_dir
    
    for key, val in kwargs.items():
        command.extend([f"--override_{key}", str(val)])
    return command

def main(config_file):
    ext = os.path.splitext(config_file)
    if ext[1] == ".py":
        with open(config_file, "r") as f:
            configs = f.read().splitlines()
            
        for i, config in enumerate(configs):
            subprocess.run(run_experiments_command(i+1, config))
    elif ext[1] == ".json":
        with open(config_file, "r") as f:
            configs = json.load(f)

        for i, config in enumerate(configs):
            subprocess.run(run_experiments_command(i+1, **config))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs curriculum learning on the datasets from the provided config files."
    )

    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default="curriculum/curriculum.py",
        help="Path to the config file which stores the list of config files to be run."
    )

    args = parser.parse_args()
    main(args.config_file)
