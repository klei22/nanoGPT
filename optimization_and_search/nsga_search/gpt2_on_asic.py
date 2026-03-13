from search_space import Individual, HeteroSearchSpace
from typing import List, Dict, Any, Tuple
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Load data again
file_path = './hw_eval/nov_arch_data/ppa_nov_core_top_extracted_data_10_1.csv'
data = pd.read_csv(file_path)

#  Creating a nested dictionary-based database
database = {}
clk_period = 5  # ns


# Populating the database with configuration as keys and relevant metrics as values
for _, row in data.iterrows():
    key = (row['MAC NUM'], row['Wmem Depth'], row['Cache Depth'], row['Clock Period (ns) Entered'])
    database[key] = {
        'power': row['Power (W)'],
        'slack': row['Clock Slack (ns)'],
        'clk_min_period': row['Clock_Min_Period'],
        'area': row['Area (um^2)']
        # Additional metrics can be added here as needed
    }

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


class Hardware_encoding:
    def __init__(self, n_col: int, n_mac: int):
        # deterministic encoding of hardware configuration
        self.n_col = n_col
        self.n_mac = n_mac

        # derived attributes
        self.n_row = None
        self.n_pe = None
        self.wmem_depth = None
        self.kvcache_depth = None

        self.ac_wmem_depth = None
        self.ac_kvcache_depth = None

    def isgpt2_feasible(self) -> bool:
        # Check if the hardware configuration is feasible for the given individual
        n_embd =  768
        # Use a mask falling back to the number of layers if L_max isn't defined
        layers = 12
        # n_head_list = [layer["n_head"] if mask[i] else 0 for i, layer in enumerate(layers)]
        n_head_list = [12 for i in range(layers)]
        n_kv_group_list = [12 for i in range(layers)]

        n_head_max = max(n_head_list)
        n_head = n_head_max
        kv_ratio_min = min([n_head_list[i] / n_kv_group_list[i] for i in range(len(n_head_list)) if n_kv_group_list[i] > 0])
        
        block_size = 1024
        head_dim = n_embd // n_head
        if n_embd % n_head != 0:
            return False
        if n_embd % self.n_col != 0:
            return False
        core_dim = n_embd // self.n_col
        if core_dim % self.n_mac != 0:
            return False
        if block_size % self.n_col != 0:
            return False

        self.n_row = n_head
        self.kvcache_depth = int(2 * n_embd * block_size / self.n_mac / self.n_col / self.n_row / kv_ratio_min)
        self.n_pe = self.n_col * self.n_row
        self.wmem_depth = int(4 * n_embd * n_embd / self.n_row / self.n_col / self.n_mac)

        if self.wmem_depth > 8192 or self.kvcache_depth > 8192:
            return False

        self.ac_kvcache_depth = sram_depth_round_up(self.kvcache_depth)
        self.ac_wmem_depth = sram_depth_round_up(self.wmem_depth)
        return True

    def get_TTFT_in_cycle(self, seq_len: int = 256) -> float:
        
        n_embd = 768
        gbus_width = self.n_mac * 8
        n_head = self.n_row
        n_cols = self.n_col
        block_size = 1024

        ttft_cycles = 0
        for i in range(0,12):
            layer_cycle = 0
            mlp_size = 3072
            mlp_ratio = mlp_size / n_embd
            layer_head = 12
            layer_kv_group = 12
            layer__kv_ratio = layer_head / layer_kv_group

            layer_cycle += (4 * n_embd * n_embd + 2 * mlp_size * n_embd) / gbus_width  # load weights on chip

            layer_cycle += ((2 * n_embd * n_embd + 2 * n_embd * n_embd / layer__kv_ratio) * seq_len + 2 * block_size * block_size * n_embd + 2 * mlp_size * n_embd * seq_len) / (n_head * n_cols * self.n_mac)  # MAC operations

            # v-link latency penalty
            if (layer_head < n_head) :
                n_groups = n_head / layer_head

                layer_cycle += seq_len * (n_groups - 1) * (n_embd / n_head) / gbus_width  # assuming perfect interleaving
                
                # mlp ratio penalty
                layer_cycle += seq_len * (mlp_ratio - 2) * n_embd / gbus_width  # assuming perfect interleaving

            # add residual delay
            layer_cycle += 2 * n_embd * seq_len / (self.n_mac * n_head * n_cols)  # load and store residual

            ttft_cycles += layer_cycle

        return ttft_cycles

    def get_token_energy_in_cycle(self) -> float:
        n_embd = 768
        gbus_width = self.n_mac * 8
        n_head = self.n_row
        n_cols = self.n_col
        block_size = 1024

        token_energy_cycles = 0
        for i in range(0,12):
            
            layer_cycle = 0
            mlp_size = 3072
            mlp_ratio = mlp_size / n_embd
            layer_head = 12
            layer_kv_group = 12
            layer__kv_ratio = layer_head / layer_kv_group

            layer_cycle += 0.2 * (4 * n_embd * n_embd + 2 * mlp_size * n_embd) / gbus_width  # load weights on chip

            layer_cycle += ((2 * n_embd * n_embd + 2 * n_embd * n_embd / layer__kv_ratio) + 2 * block_size * block_size * n_embd + 2 * mlp_size * n_embd) / (n_head * n_cols * self.n_mac)  # MAC operations

            # v-link latency penalty
            if (layer_head < n_head) :
                n_groups = n_head / layer_head

                layer_cycle += 0.2 * (n_groups - 1) * (n_embd / n_head) / gbus_width  # assuming perfect interleaving
                
                # mlp ratio penalty
                layer_cycle += 0.2 * (mlp_ratio - 2) * n_embd / gbus_width  # assuming perfect interleaving

            # add residual delay
            layer_cycle += 2 * n_embd / (self.n_mac * n_head * n_cols)  # load and store residual

            token_energy_cycles += layer_cycle

        return token_energy_cycles

# Hardware stat dataclass
class HardwareStat:
    def __init__(self, params_m: float, mem_gb: float, energy_per_token: float, ttft: float):
        self.energy_per_token = energy_per_token
        self.ttft = ttft
        self.n_sram_access = None
        self.sram_rb = None
        self.sram_wb = None

# def evaluate_hardware(ind: Individual) -> Dict[str, Any]:
    # First map the individual to feasiable hardware configuration

def sram_depth_round_up(depth: int) -> int:
    # Round up to nearest power of 2
    if depth <= 128:
        return 128
    elif depth <= 1024:
        # round up to nearest 256
        return int((depth + 255) // 256 * 256)
    elif depth <= 4096:
        # round up to nearest 512
        return int((depth + 511) // 512 * 512)  
    else:
        # round up to nearest 1024
        return int((depth + 1023) // 1024 * 1024)
    
def evaluate_gpt2_on_hardware() -> Dict[str, Any]:
    ind_results = []
    for col in range(1, 33):  # n_col from 1 to 32:
        for mac in [4, 8, 16, 32]:
            hw = Hardware_encoding(n_col=col, n_mac=mac)
            if hw.isgpt2_feasible():
                energy = hw.get_token_energy_in_cycle()
                ttft = hw.get_TTFT_in_cycle()

                core_power = database.get((hw.n_mac, hw.ac_wmem_depth, hw.ac_kvcache_depth, clk_period), {}).get('power', 'N/A')  # assuming clock period 5ns
                core_area = database.get((hw.n_mac, hw.ac_wmem_depth, hw.ac_kvcache_depth, clk_period), {}).get('area', 'N/A')
                if core_power != 'N/A' and core_area != 'N/A':
                    n_head = hw.n_row
                    n_cols = hw.n_col
                    total_area = core_area * n_head * n_cols
                    total_power = core_power * n_head * n_cols

                    ttft_ns = ttft * clk_period  # assuming clock period 5ns
                    energy_per_token_mJ = energy * clk_period * total_power / 1e6  # convert to mJ

                    ind_results.append({
                        "n_rows": n_head,
                        "n_cols": n_cols,
                        "n_mac": hw.n_mac,
                        "total_area_um2": total_area,
                        "total_power_W": total_power,
                        "ttft_cycles": ttft,
                        "ttft_ns": ttft_ns,
                        "energy_per_token_cycles": energy,
                        "energy_per_token_mJ": energy_per_token_mJ,
                    })

    # pick the Pareto front (first front) configurations for this individual
    if not ind_results:
        return {
                    "n_rows": 0,
                    "n_cols": 0,
                    "n_mac": 0,
                    "total_area_um2": float('inf'),
                    "total_power_W": float('inf'),
                    "ttft_cycles": float('inf'),
                    "ttft_ns": float('inf'),
                    "energy_per_token_cycles": float('inf'),
                    "energy_per_token_mJ": float('inf'),
                }
    df_ind = pd.DataFrame(ind_results)
    if not df_ind.empty:
        points = df_ind[['energy_per_token_cycles', 'ttft_cycles']].to_numpy()
        keep = np.ones(len(points), dtype=bool)
        for i, (x, y) in enumerate(points):
            if not keep[i]:
                continue
            dominated = (points[:, 0] <= x) & (points[:, 1] <= y) & ((points[:, 0] < x) | (points[:, 1] < y))
            dominated[i] = False
            keep &= ~dominated
        pareto_front = df_ind.loc[keep].sort_values(by=['energy_per_token_cycles', 'ttft_cycles'])
        # report when pareto_front is more than one point
        if len(pareto_front) > 1:
            print("Pareto front configurations for this individual:")
            print(pareto_front)
        
        return pareto_front.iloc[0].to_dict()

if __name__ == "__main__":
    
    print(evaluate_gpt2_on_hardware())



# add a filter to only keep pareto-optimal configurations and obeys the constraints
