from nsga2 import Population
from typing import List, Dict, Any
from search_space import Individual
from search_space import HeteroSearchSpace
import yaml
from remote_trainer import RemoteTrainer  
import logging
import time
import os


population = Population.load_checkpoint("ckpts/0929_2359_gen3.json", from_pkl=False)
max_n_layer = 10
search_space = HeteroSearchSpace(L_max=max_n_layer)

population.search_space = search_space

population.print_summary()

population.save_checkpoint_pkl("ckpts/929_gen3.pkl")



