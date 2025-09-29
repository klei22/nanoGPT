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

#initialize Population class from nsga.py with individuals randomly
init_population_size = 32
max_n_layer = 10
search_space = HeteroSearchSpace(L_max=max_n_layer)
individuals = [search_space.sample() for _ in range(init_population_size)]
population = Population(individuals, search_space=search_space)
population.delete_duplicates()  # Remove duplicates if any


# TODO: Define remote training parameters
hosts = ["34.85.168.66", "34.132.101.194"]  # Instance IP addresses
user = "xinting"  # SSH username for login
key_filename = "/home/xinting/.ssh/id_rsa"  # Path to SSH private key file

population.sw_eval(hosts=hosts, user=user, key_filename=key_filename)
population.print_summary()

# nsga parameters defined here
population.n_population = init_population_size
population.n_offspring = 16

run_time = time.strftime("%m%d_%H%M", time.localtime())
n_gen = 15
for gen in range(1, n_gen + 1):
    print(f"\n\n================ Generation {gen} ================\n")
    population.generate_offspring()
    population.sw_eval(hosts=hosts, user=user, key_filename=key_filename)
    population.update_elimination()
    population.print_summary()
    checkpoint_filename = f"ckpts/{run_time}_gen{gen}.json"
    # create directory if not exists
    os.makedirs(os.path.dirname(checkpoint_filename), exist_ok=True)
    population.save_checkpoint(checkpoint_filename)
    print(f"Checkpoint saved to {checkpoint_filename}")

population.save_checkpoint_pkl(f"ckpts/{run_time}_final_pop_gen{n_gen}.pkl")




