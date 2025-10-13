from nsga2 import Population
from typing import List, Dict, Any
from search_space import Individual
from search_space import HeteroSearchSpace
import yaml
from remote_trainer import RemoteTrainer  
import logging
import time
from run_exp import load_hosts_from_file
import os


# Configure logging to only show INFO:root messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')
# Disable all other loggers except root
for name in ("paramiko", "paramiko.transport", "fabric", "invoke"):
    logging.getLogger(name).disabled = True

hosts = load_hosts_from_file("../host_configs/internal_hosts.yaml")
user = "xinting"  # SSH username for login
key_filename = "/home/xinting/.ssh/id_rsa"  # Path to SSH private key file

trainer = RemoteTrainer(hosts=hosts, user=user, key_filename=key_filename)
trainer.clear_all_jobs()
print("Cleared all training jobs on remote hosts.")



