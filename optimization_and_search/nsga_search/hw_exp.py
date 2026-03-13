# hardware exploration by TimeLoop

from unittest import result
from search_space import Individual
import yaml
import os
import time
import timeloopfe.v4 as tl
from utils.parse_timeloop_stats import parse_timeloop_stats
import matplotlib.pyplot as plt
import csv
from pathlib import Path

# Define relative paths
ARCH_PATH = f"{os.curdir}/hw_eval/arch/system_gemmini.yaml"
COMPONENTS_PATH = f"{os.curdir}/hw_eval/arch/components/*.yaml"
PROBLEM_PATH = f"{os.curdir}/hw_eval/prob/generic_GEMM.yaml"
MAPPER_PATH = f"{os.curdir}/hw_eval/mapper/mapper.yaml"
CONSTRAINTS_PATH = f"{os.curdir}/hw_eval/constraints/constraints.yaml"
VARIABLES_PATH = f"{os.curdir}/hw_eval/mapper/variables.yaml"

def run_GEMM_evaluation(in_channel: int, out_channel: int, seq_length: int, work_dir: str, log_path: str = "/tmp/timeloop.log") -> dict:
    
    # create working directory if not exists
    os.makedirs(work_dir, exist_ok=True)
    # Prepare problem file with specific dimensions
    out_dir = os.path.join(work_dir, f"gemm_{in_channel}i_{out_channel}o_{seq_length}l")
    os.makedirs(out_dir, exist_ok=True)
    problem_file = os.path.join(out_dir, "generic_GEMM.yaml")
    with open(PROBLEM_PATH, 'r') as f:
        problem_data = f.read()
        problem_data = problem_data.replace("$IN_CHANNELS", str(in_channel))
        problem_data = problem_data.replace("$OUT_CHANNELS", str(out_channel))
        problem_data = problem_data.replace("$OUT_HEIGHT", str(seq_length))
    with open(problem_file, 'w') as f:
        f.write(problem_data)

    spec = tl.Specification.from_yaml_files(
        ARCH_PATH,
        COMPONENTS_PATH,
        MAPPER_PATH,
        problem_file,
        CONSTRAINTS_PATH,
        VARIABLES_PATH
    )

    spec.mapspace.template = 'uber' #'ruby'
    constrained_factors = ["D=1"]
    constrained_factors.append("E=1")
    tl.constraints.Factors(constrained_factors)
    if spec.constraints['targets'] is None:
        spec.constraints['targets'] = tl.constraints.ConstraintsList()

    output_file = os.path.join(out_dir, f"timeloop-mapper.stats.txt")
    if not os.path.exists(output_file):
        # Run the Timeloop mapper
        if not os.path.exists(log_path):
            with open(log_path, 'w') as f:
                f.write("")  # create the log file
        tl.call_mapper(spec, output_dir=out_dir, log_to=log_path)

    return parse_timeloop_stats(output_file)

def evaluate_layer(layer: dict, n_embd: int, seq_length: int, work_dir: str) -> dict:
    try:
        n_head = layer['n_head']
        n_kv_groups = layer['n_kv_group'] 
        n_qk_head_dim = layer['n_qk_head_dim']
        n_v_head_dim = layer['n_v_head_dim']
        # use_concat_heads = layer['use_concat_heads']
        n_cproj = layer['n_cproj']
        attn_variant = layer['attention_variant']
        mlp_size = layer['mlp_size']
    except KeyError as e:
        raise KeyError(f"Missing key in layer definition: {e}")
    
    # Initialize TimeLoop in parallel mode
    if attn_variant == 'infinite':
        #1. QK GEN
        qk_gen_stats = run_GEMM_evaluation(in_channel=n_embd, out_channel=n_qk_head_dim * (n_head + n_kv_groups), seq_length=seq_length, work_dir=work_dir)
        #2. V GEN
        v_gen_stats = run_GEMM_evaluation(in_channel=n_embd, out_channel=n_v_head_dim * n_kv_groups, seq_length=seq_length, work_dir=work_dir)
        #3. QK ATTN
        qk_attn_stats = run_GEMM_evaluation(in_channel=n_qk_head_dim, out_channel=seq_length, seq_length=n_head // n_kv_groups, work_dir=work_dir)
        # scale this by n_kv_groups
        qk_attn_stats['cycles'] = qk_attn_stats['cycles'] * n_kv_groups if qk_attn_stats['cycles'] is not None else None
        qk_attn_stats['energy_uJ'] = qk_attn_stats['energy_uJ'] * n_kv_groups if qk_attn_stats['energy_uJ'] is not None else None
        qk_attn_stats['total_ops'] = qk_attn_stats['total_ops'] * n_kv_groups if qk_attn_stats['total_ops'] is not None else None
        qk_attn_stats['total_memory_accesses'] = qk_attn_stats['total_memory_accesses'] * n_kv_groups if qk_attn_stats['total_memory_accesses'] is not None else None
        #4. PV ATTN
        pv_attn_stats = run_GEMM_evaluation(in_channel=seq_length, out_channel=n_v_head_dim, seq_length=n_head // n_kv_groups, work_dir=work_dir)
        # scale this by n_kv_groups
        pv_attn_stats['cycles'] = pv_attn_stats['cycles'] * n_kv_groups if pv_attn_stats['cycles'] is not None else None
        pv_attn_stats['energy_uJ'] = pv_attn_stats['energy_uJ'] * n_kv_groups if pv_attn_stats['energy_uJ'] is not None else None
        pv_attn_stats['total_ops'] = pv_attn_stats['total_ops'] * n_kv_groups if pv_attn_stats['total_ops'] is not None else None
        pv_attn_stats['total_memory_accesses'] = pv_attn_stats['total_memory_accesses'] * n_kv_groups if pv_attn_stats['total_memory_accesses'] is not None else None
        #5. ATTN PROJ
        attn_proj_stats = run_GEMM_evaluation(in_channel=n_v_head_dim*n_head, out_channel=n_embd, seq_length=seq_length, work_dir=work_dir)
    
    #6. MLP FC1
    mlp_fc1_stats = run_GEMM_evaluation(in_channel=n_embd, out_channel=mlp_size, seq_length=seq_length, work_dir=work_dir)
    #7. MLP FC2
    mlp_fc2_stats = run_GEMM_evaluation(in_channel=mlp_size, out_channel=n_embd, seq_length=seq_length, work_dir=work_dir)

    # Aggregate all layer stats
    if attn_variant == 'infinite':
        layer_stats = aggregate_stats([
            qk_gen_stats,
            v_gen_stats,
            qk_attn_stats,
            pv_attn_stats,
            attn_proj_stats,
            mlp_fc1_stats,
            mlp_fc2_stats
        ])
    else:
        layer_stats = aggregate_stats([
            mlp_fc1_stats,
            mlp_fc2_stats
        ])
    return layer_stats

def eval_individual(individual: Individual, work_dir: str) -> dict:
    global_spec = individual["globals"]
    layer_spec = individual["layers"]
    n_embd = global_spec["n_embd"]
    seq_length = global_spec["block_size"]
    layer_mask = global_spec.get("layer_mask", None)
    if layer_mask is None:
        raise ValueError("layer_mask is not defined in global_spec")

    hw_eval_list = []
    for i, layer in enumerate(layer_spec):
        if layer_mask[i] == 1:
            layer_stats = evaluate_layer(layer, n_embd, seq_length, work_dir)
            hw_eval_list.append(layer_stats)

    aggregated_stats = aggregate_stats(hw_eval_list)
        
    # average over sequence length
    aggregated_stats['cycles_per_token'] = aggregated_stats['cycles'] / seq_length if aggregated_stats['cycles'] is not None else None
    aggregated_stats['token_delay'] = aggregated_stats['cycles_per_token'] / 1e9  # assuming 1GHz clock
    aggregated_stats['energy_per_token_uJ'] = aggregated_stats['energy_uJ'] / seq_length if aggregated_stats['energy_uJ'] is not None else None
    aggregated_stats['edp_per_token'] = aggregated_stats['edp'] / seq_length if aggregated_stats['edp'] is not None else None
    return aggregated_stats

def evaluate_population(population: list, base_work_dir: str) -> list:
    results = []
    for i, individual in enumerate(population):
        print(f"Evaluating Individual {i}...")
        individual_stats = eval_individual(individual, work_dir=base_work_dir)
        results.append(individual_stats)
    
    return results

def aggregate_stats(stats_list: list) -> dict:
    aggregated_stats = {}
    for key in stats_list[0].keys():
        aggregated_stats[key] = sum(
            stats[key] for stats in stats_list if stats[key] is not None
        )

    # recalculate derived metrics
    if aggregated_stats['total_ops'] is not None and aggregated_stats['total_memory_accesses'] is not None and aggregated_stats['total_memory_accesses'] != 0:
        aggregated_stats['algorithmic_intensity_ops_per_access'] = aggregated_stats['total_ops'] / aggregated_stats['total_memory_accesses']
    else:
        aggregated_stats['algorithmic_intensity_ops_per_access'] = None
    aggregated_stats['algorithmic_intensity_ops_per_byte'] = aggregated_stats['algorithmic_intensity_ops_per_access']
    aggregated_stats['edp'] = aggregated_stats['energy_uJ'] * aggregated_stats['cycles'] / 10e6 if aggregated_stats['energy_uJ'] is not None and aggregated_stats['cycles'] is not None else None  # J*ns
    
    total_cycle = aggregated_stats['cycles']
    aggregated_stats['utilization_pct'] = 0
    aggregated_stats['gflops'] = 0
    for stats in stats_list:
        aggregated_stats['utilization_pct'] += (stats['utilization_pct'] * stats['cycles'] / total_cycle) if stats['utilization_pct'] is not None and stats['cycles'] is not None else 0
        aggregated_stats['gflops'] += (stats['gflops'] * stats['cycles'] / total_cycle) if stats['gflops'] is not None and stats['cycles'] is not None else 0

    return aggregated_stats

