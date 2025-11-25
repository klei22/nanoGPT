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

# Configure logging to only show INFO:root messages
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


def load_search_space_from_yaml(path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Search space file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Search space YAML must define a mapping with 'global_spec' and 'layer_spec'.")

    global_spec = data.get("global_spec")
    layer_spec = data.get("layer_spec")

    if not isinstance(global_spec, dict) or not isinstance(layer_spec, dict):
        raise ValueError("Search space YAML missing 'global_spec' or 'layer_spec' dictionaries.")

    return global_spec, layer_spec


def main():
    parser = argparse.ArgumentParser(description="Run NSGA-II search with remote evaluation")
    parser.add_argument(
        "--hosts-file",
        type=str,
        default="../host_configs/hosts.yaml",
        help="Path to a YAML hosts file containing a top-level list of IPs",
    )
    parser.add_argument("--user", type=str, default="xinting", help="SSH username")
    parser.add_argument("--key", type=str, default="/home/xinting/.ssh/id_rsa", help="Path to SSH private key")
    parser.add_argument("--pop_size", type=int, default=16, help="Population size")
    parser.add_argument("--max_layers", type=int, default=10, help="Max number of layers (L_max)")
    parser.add_argument("--min_layers", type=int, default=1, help="Min number of layers (L_min)")
    parser.add_argument("--offspring", type=int, default=8, help="Number of offspring per generation")
    parser.add_argument("--generations", type=int, default=15, help="Number of generations to run")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="Path to checkpoint file to resume from (optional)")
    parser.add_argument("--exp_name", type=str, default="infi_attn_exp_iter20k", help="Experiment name for checkpoint directory")
    parser.add_argument("--conda_env", type=str, default="reallmforge", help="Conda environment name on remote hosts")
    parser.add_argument("--max_iters", type=int, default=10000, help="Max training iterations per evaluation")
    parser.add_argument("--crossover_rate", type=float, default=0.9, help="Crossover rate for NSGA-II")
    parser.add_argument("--mutation_rate", type=float, default=0.1, help="Mutation rate for NSGA-II")
    parser.add_argument(
        "--search_space_config",
        type=str,
        default="search_space_def/default_search_space.yaml",
        help="Path to YAML file defining 'global_spec' and 'layer_spec' (relative paths resolve from this script)",
    )
    args = parser.parse_args()

    # set random seed for reproducibility
    random.seed(65)

    hosts = load_hosts_from_file(args.hosts_file)
    logging.info(f"Loaded {len(hosts)} hosts from {args.hosts_file}")
    user = args.user
    key_filename = args.key

    init_population_size = args.pop_size
    max_n_layer = args.max_layers
    min_n_layer = args.min_layers
    config_path = args.search_space_config
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)

    global_spec, layer_spec = load_search_space_from_yaml(config_path)
    search_space = HeteroSearchSpace.from_dicts(global_spec, layer_spec, L_max=max_n_layer, L_min=min_n_layer)
    
    print("Using search space:")
    print(search_space.print_search_space())

    exp_name = args.exp_name

    objs = ["val_loss", "energy_per_token_mJ", "ttft_ns"]  # Minimize validation loss and number of parameters
    cons = {
        "params": 800_000_000,  # 800 million params
        "val_loss": 3.6,  # 3.6
        }

    # initial evaluation
    if args.resume_ckpt is not None:
        if os.path.exists(args.resume_ckpt):
            logging.info(f"Resuming from checkpoint: {args.resume_ckpt}")
            population = Population.load_checkpoint(args.resume_ckpt, from_pkl=args.resume_ckpt.endswith('.pkl'))
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {args.resume_ckpt}")
        population.search_space = search_space  # Ensure search space is set

        population.objs_settings = objs
        population.cons_settings = cons
        population.print_summary()
    else:
        # initialize Population class from nsga.py with individuals randomly
        individuals = [search_space.sample() for _ in range(init_population_size)]
        

        population = Population(individuals, search_space=search_space, objs_settings=objs, cons_settings=cons)
        population.delete_duplicates()  # Remove duplicates if any

        # initial evaluation
        population.sw_eval(hosts=hosts, user=user, key_filename=key_filename, run_dir_name=exp_name, conda_env=args.conda_env, max_iters=args.max_iters, sw_only=True, hw_eval_on_reallmasic=True)
        population.print_summary()

    # nsga parameters defined here
    population.n_population = init_population_size
    population.n_offspring = args.offspring
    population.crossover_rate = args.crossover_rate
    population.mutation_rate = args.mutation_rate

    # save initial checkpoint
    run_time = time.strftime("%m%d_%H%M", time.localtime())
    if args.resume_ckpt is None:
        population.save_checkpoint(f"ckpts/{exp_name}/{run_time}_ckpt_gen{population.gen}.json")
        population.save_checkpoint_pkl(f"ckpts/{exp_name}/pkl/{run_time}_pop_gen{population.gen}.pkl")

    # update the working directory on remote hosts
    trainer = RemoteTrainer(hosts=hosts, user=user, key_filename=key_filename)
    trainer.perform_git_pull(remote_work_dir=f"/home/{user}/Evo_GPT")

    # run_time = time.strftime("%m%d_%H%M", time.localtime())
    n_gen = args.generations
    for i in range(0, n_gen):
        population.generate_offspring()
        gen = population.gen
        print(f"\n\n================ Generation {gen} ================\n")
        population.sw_eval(hosts=hosts, user=user, key_filename=key_filename, run_dir_name=exp_name, conda_env=args.conda_env, max_iters=args.max_iters, sw_only=True, hw_eval_on_reallmasic=True)
        population.save_checkpoint(f"ckpts/{exp_name}/{run_time}_ckpt_offspring_gen{gen}.json")
        population.update_elimination()
        population.print_summary()
        population.save_checkpoint(f"ckpts/{exp_name}/{run_time}_ckpt_gen{gen}.json")
        population.save_checkpoint_pkl(f"ckpts/{exp_name}/pkl/{run_time}_pop_gen{gen}.pkl")

if __name__ == "__main__":
    main()









