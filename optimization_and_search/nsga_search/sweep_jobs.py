from nsga2 import Population
from typing import List, Dict, Any, Tuple
from search_space import Individual
from search_space import HeteroSearchSpace
import yaml
from remote_trainer import RemoteTrainer  
import logging
import time
import os
import argparse
import random

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')

# Disable all other loggers except root
for name in ("paramiko", "paramiko.transport", "fabric", "invoke"):
    logging.getLogger(name).disabled = True

def load_hosts_from_file(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Hosts file not found: {path}")
    hosts: List[str] = []
    _, ext = os.path.splitext(path)
    try:
        if ext.lower() not in (".yaml", ".yml"):
            raise ValueError("Hosts file must be a YAML file (.yaml or .yml) with a top-level list of IPs")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, list):
            raise ValueError("Hosts YAML must be a top-level list, e.g.\n- 1.2.3.4\n- 5.6.7.8")

        hosts = [str(x).strip() for x in data if isinstance(x, (str, int, float)) and str(x).strip()]
    except Exception as e:
        raise RuntimeError(f"Failed to parse hosts file '{path}': {e}")

    if not hosts:
        raise ValueError(f"No hosts parsed from file: {path}")
    return hosts

def main():
    parser = argparse.ArgumentParser(description="Run grid search on given yaml configurations")
    parser.add_argument(
        "--hosts-file",
        type=str,
        default="../host_configs/host_no_east4.yaml",
        help="Path to a YAML hosts file containing a top-level list of IPs",
    )
    parser.add_argument("--user", type=str, default="xinting", help="SSH username")
    parser.add_argument("--key", type=str, default="/home/xinting/.ssh/id_rsa", help="Path to SSH private key")
    parser.add_argument("--conda_env", type=str, default="reallmforge", help="Conda environment name on remote hosts")
    parser.add_argument("--config_yaml", type=str, default="tests/grid_mlp_placement_smol_refactor.yaml", help="Path to grid search configuration YAML")
    parser.add_argument("--run_dir_name", type=str, default="run_sweep_mlp", help="Run directory name")

    parser.add_argument("--max_iters", type=int, default=10000, help="Max training iterations per evaluation")

    args = parser.parse_args()

    hosts = load_hosts_from_file(args.hosts_file)
    logging.info(f"Loaded {len(hosts)} hosts from {args.hosts_file}")
    user = args.user
    key_filename = args.key

    trainer = RemoteTrainer(hosts=hosts, user=user, key_filename=key_filename)
    trainer.perform_git_pull(remote_work_dir=f"/home/{user}/Evo_GPT")

    trainer.submit_job(path_to_yaml=args.config_yaml, remote_work_dir=f"/home/{user}/Evo_GPT", dir_name=args.run_dir_name, max_iters=args.max_iters, conda_env=args.conda_env)
    time.sleep(5)  # wait a bit before polling
    trainer.poll_jobs() 

    trainer.wait_for_all(poll_interval=120, timeout=100000, verbose=True)
    data_csv = trainer.fetch_results(local_dir="train", gen=None)

    print(f"Fetched results to {data_csv}")

    # sw_data = load_csv_with_idx_lookup(data_csv)
    # print (f"Loaded {len(sw_data)} results from {data_csv}")




if __name__ == "__main__":
    main()