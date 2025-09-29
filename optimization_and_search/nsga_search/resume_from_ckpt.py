from nsga2 import Population
from typing import List, Dict, Any
from search_space import Individual
from search_space import HeteroSearchSpace
import yaml
from remote_trainer import RemoteTrainer  
import logging
import time

# Configure logging to only show INFO:root messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')
# Disable all other loggers except root
for name in ("paramiko", "paramiko.transport", "fabric", "invoke"):
    logging.getLogger(name).disabled = True

# load from checkpoint
population = Population.load_checkpoint("ckpts/init_pop.pkl")

print("Loaded population from checkpoint.")
population.print_summary()

hosts = ["34.58.151.71", "34.85.168.66", "34.11.48.206", "34.86.55.236", "35.245.124.235"]
# hosts = ["34.85.168.66", "34.11.48.206", "34.86.55.236"]
user = "xinting"
key_filename = "/home/xinting/.ssh/id_rsa"

population.gen = 1
population.n_offspring = 10

# population.generate_offspring()
# population.print_details()
# population.sw_eval(hosts=hosts, user=user, key_filename=key_filename)

# population.save_checkpoint("ckpts/gen1_pop.json")

run_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
# use time for name

n_gen = 5
for gen in range(2, n_gen + 1):
    print(f"\n\n================ Generation {gen} ================\n")
    population.generate_offspring()
    population.sw_eval(hosts=hosts, user=user, key_filename=key_filename)
    population.update_elimination()
    population.print_summary()
    population.save_checkpoint(f"ckpts/{run_time}_ckpt_gen{gen}.json")

population.save_checkpoint_pkl(f"ckpts/{run_time}_final_pop_gen{n_gen}.pkl")

# population.sw_eval(hosts=hosts, user=user, key_filename=key_filename)



