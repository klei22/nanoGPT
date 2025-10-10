from nsga2 import Population
from typing import List, Dict, Any
from search_space import Individual
from search_space import HeteroSearchSpace
import yaml
from remote_trainer import RemoteTrainer  
import logging
import time
import os


# Configure logging to only show INFO:root messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')
# Disable all other loggers except root
for name in ("paramiko", "paramiko.transport", "fabric", "invoke"):
    logging.getLogger(name).disabled = True

init_population_size = 16
max_n_layer = 16
min_n_layer = 4

# fix n_embd and block_size to reduce search space
global_spec = {
        "n_embd": {"type": "int", "low": 768, "high": 768, "step": 128},
        "block_size": {"type": "int", "low": 512, "high": 512, "step": 128},
        "use_concat_heads": {"type": "cat", "choices": [True, False]}
    }

layer_spec = {
        "n_head": {"type": "int", "low": 1, "high": 16, "step": 1},
        "n_kv_group": {"type": "int", "low": 1, "high": 16, "step": 1},
        "mlp_size": {"type": "int", "low": 256, "high": 4096, "step": 256},
        "n_qk_head_dim": {"type": "int", "low": 16, "high": 128, "step": 16},
        "n_v_head_dim": {"type": "int", "low": 16, "high": 128, "step": 16},
        "n_cproj": {"type": "int", "low": 1, "high": 4, "step": 1},
        "attention_variant": {"type": "cat", "choices": ["infinite", "identity"]},
    }

search_space = HeteroSearchSpace.from_dicts(global_spec, layer_spec, L_max=max_n_layer, L_min=min_n_layer)

individuals = [search_space.sample() for _ in range(init_population_size)]

population = Population(individuals, search_space=search_space)
population.delete_duplicates()  # Remove duplicates if any

hosts = ["34.11.48.206", "34.86.55.236", "35.245.124.235", "34.144.183.145", "34.85.252.132", "34.162.10.11", "34.11.9.189", "34.85.233.248"]  # Instance IP addresses
user = "xinting"  # SSH username for login
key_filename = "/home/xinting/.ssh/id_rsa"  # Path to SSH private key file

trainer = RemoteTrainer(hosts=hosts, user=user, key_filename=key_filename)
trainer.perform_git_pull(remote_work_dir=f"/home/{user}/Evo_GPT")

exp_name = "infi_attn_exp_2"

population.sw_eval(hosts=hosts, user=user, key_filename=key_filename, run_dir_name=exp_name)
population.print_summary()

# nsga parameters defined here
population.n_population = init_population_size
population.n_offspring = 8

run_time = time.strftime("%m%d_%H%M", time.localtime())
population.save_checkpoint(f"ckpts/{exp_name}/{run_time}_ckpt_gen{population.gen}.json")
population.save_checkpoint_pkl(f"ckpts/{exp_name}/{run_time}_pop_gen{population.gen}.pkl")

n_iter = 30
for i in range(0, n_iter):
    population.generate_offspring()
    gen = population.gen
    print(f"\n\n================ Generation {gen} ================\n")
    population.sw_eval(hosts=hosts, user=user, key_filename=key_filename)
    population.update_elimination()
    population.print_summary()
    population.save_checkpoint(f"ckpts/{exp_name}/{run_time}_ckpt_gen{gen}.json")
    population.save_checkpoint_pkl(f"ckpts/{exp_name}/{run_time}_pop_gen{gen}.pkl")

