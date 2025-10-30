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
from hw_exp import *
from nsga2 import Population


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
    results = evaluate_population(population.individuals, base_work_dir=base_runs_dir)
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
            print(r)

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