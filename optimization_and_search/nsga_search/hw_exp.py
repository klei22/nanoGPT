# hardware exploration by TimeLoop

from unittest import result
from search_space import Individual
from nsga2 import Population
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

def eval_individual(individual: Individual, seq_length: int, work_dir: str) -> dict:
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
        
    return aggregate_stats(hw_eval_list)

def evaluate_population(population: Population, seq_length: int, base_work_dir: str, eval_off_spring: bool = False) -> list:
    results = []
    if eval_off_spring:
        for i, individual in enumerate(population.individuals):
            print(f"Evaluating Individual {i}...")
            individual_stats = eval_individual(individual, seq_length, work_dir=base_work_dir)
            results.append(individual_stats)
    else:
        print("Evaluating Parents Only...")
        for i, individual in enumerate(population.individuals):
            print(f"Evaluating Individual {i}...")
            individual_stats = eval_individual(individual, seq_length, work_dir=base_work_dir)
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

def main():
    # layer = {
    #     'n_head': 8,
    #     'n_kv_groups': 4,
    #     'n_qk_head_dim': 64,
    #     'n_v_head_dim': 64,
    #     'use_concat_heads': False,
    #     'n_cproj': 512,
    #     'attn_variant': 'infinite',
    #     'mlp_size': 2048
    # }
    # n_embd = 512
    # seq_length = 128
    # work_dir = "./hw_eval/runs/"
    # result = evaluate_layer(layer, n_embd, seq_length, work_dir)
    # print("\n Aggregated Layer Stats:", result)

    population = Population.load_checkpoint("/home/xinting/Evo_GPT/optimization_and_search/nsga_search/ckpts/infi_medium/ckpt_gen101.json", from_pkl=False)

    print("Evaluation started...")
    time_start = time.time()
    # Run evaluations in parallel for the whole population
    base_runs_dir = "./hw_eval/runs/"
    cur_dir = "./"
    results = evaluate_population(population, seq_length=128, base_work_dir=base_runs_dir)
    # results = eval_individual(population.individuals[12], seq_length=128, work_dir=base_runs_dir)
    time_end = time.time()
    print(f"Parallel evaluation completed in {time_end - time_start:.2f} seconds.")

    # Print per-individual results (keep compact)
    for i, r in enumerate(results):
        print(f"\n--- Individual {i} result ---")
        if isinstance(r, dict) and "error" in r:
            print(f"  ERROR: {r['error']}")
        else:
            # a short summary: cycles, total_ops, energy, utilization
            cycles = r.get('cycles') if r else None
            ops = r.get('total_ops') if r else None
            energy = r.get('energy_uJ') if r else None
            util = r.get('utilization_pct') if r else None
            print(f"  cycles={cycles}, total_ops={ops}, energy_uJ={energy}, utilization_pct={util}")

    # Create plots directory
    plots_dir = Path(cur_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Prepare scatter data: energy_uJ (x) vs cycles (y)
    x = []  # energy_uJ
    y = []  # cycles
    labels = []
    for i, r in enumerate(results):
        if not isinstance(r, dict) or 'error' in r:
            continue
        energy = r.get('energy_uJ')
        cycles = r.get('cycles')
        if energy is None or cycles is None:
            continue
        x.append(energy)
        y.append(cycles)
        labels.append(str(i))

    if x and y:
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(x, y, c='tab:blue', edgecolors='k')
        ax.set_xlabel('Energy (uJ)')
        ax.set_ylabel('Cycles')
        ax.set_title('Per-individual Energy vs Cycles')
        ax.grid(True, linestyle='--', alpha=0.4)

        # annotate points with individual index
        for xi, yi, lab in zip(x, y, labels):
            ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(4,3), fontsize=8)

        out_png = plots_dir / 'energy_vs_cycles.png'
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

        # also write CSV for easy inspection
        csv_path = plots_dir / 'energy_vs_cycles.csv'
        with open(csv_path, 'w', newline='') as csvf:
            writer = csv.writer(csvf)
            writer.writerow(['individual_idx', 'energy_uJ', 'cycles'])
            for lab, xi, yi in zip(labels, x, y):
                writer.writerow([lab, xi, yi])

        print(f"Saved scatter to {out_png} and data to {csv_path}")
    else:
        print("No valid energy/cycles data found to plot.")
    # plot results in scatters   
    

if __name__ == "__main__":
    main()