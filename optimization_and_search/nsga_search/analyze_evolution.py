#!/usr/bin/env python3
"""
Complete example of using the EvolutionHistory parser with plotting
"""

import os
from visualization.evolution_history import LogParser, EvolutionHistory

def main():
    log_file = "/home/xinting/Evo_GPT/optimization_and_search/nsga_search/logs/run_0930.log"
    
    # Create output directory for plots
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    print("Parsing evolution log file...")
    
    # Parse the log file
    parser = LogParser()
    history = parser.parse_log_file(log_file)
    
    print(f"Successfully parsed {len(history.generations)} generations")
    
    # Generate and save plots
    print("\nGenerating evolution plots...")
    
    # Plot objectives evolution
    history.plot_objective_evolution(
        objectives=['validation_loss', 'energy_per_token', 'ttft'],
        save_path=f"{plots_dir}/objectives_evolution.png"
    )
    print(f"Saved objectives evolution plot to {plots_dir}/objectives_evolution.png")
    
    # Plot constraint violations (even though they're all zero)
    history.plot_constraint_evolution(save_path=f"{plots_dir}/constraints_evolution.png")
    print(f"Saved constraints evolution plot to {plots_dir}/constraints_evolution.png")
    
    # Print detailed analysis
    print("\n" + "="*60)
    print("DETAILED EVOLUTION ANALYSIS")
    print("="*60)
    
    generations = history.get_generation_numbers()
    print(f"Generations analyzed: {generations}")
    
    # Analyze each objective
    objectives = ['validation_loss', 'energy_per_token', 'ttft']
    stats = ['min', 'avg', 'max']
    
    for obj in objectives:
        print(f"\n{obj.replace('_', ' ').upper()} EVOLUTION:")
        print("-" * 40)
        for stat in stats:
            values = history.get_objective_evolution(obj, stat)
            improvement = values[0] - values[-1]
            improvement_pct = (improvement / values[0]) * 100 if values[0] != 0 else 0
            print(f"  {stat.capitalize():>3}: {values[0]:.3f} → {values[-1]:.3f} "
                  f"(Δ: {improvement:+.3f}, {improvement_pct:+.1f}%)")
    
    # Show best and worst individuals per generation (based on average)
    print(f"\nBEST PERFORMING GENERATIONS:")
    print("-" * 40)
    
    val_loss_avgs = history.get_objective_evolution('validation_loss', 'avg')
    energy_avgs = history.get_objective_evolution('energy_per_token', 'avg')
    ttft_avgs = history.get_objective_evolution('ttft', 'avg')
    
    best_val_loss_gen = generations[val_loss_avgs.index(min(val_loss_avgs))]
    best_energy_gen = generations[energy_avgs.index(min(energy_avgs))]
    best_ttft_gen = generations[ttft_avgs.index(min(ttft_avgs))]
    
    print(f"  Best Validation Loss: Generation {best_val_loss_gen} (avg: {min(val_loss_avgs):.3f})")
    print(f"  Best Energy/Token:    Generation {best_energy_gen} (avg: {min(energy_avgs):.3f})")
    print(f"  Best TTFT:           Generation {best_ttft_gen} (avg: {min(ttft_avgs):.1f})")
    
    # Summary statistics
    summary = history.summary_stats()
    print(f"\nOVERALL SUMMARY:")
    print("-" * 40)
    print(f"Total evolution span: {summary['total_generations']} generations")
    print(f"Population size: {summary['final_population_size']} individuals")
    
    # Check for convergence (small improvements in recent generations)
    if len(val_loss_avgs) >= 3:
        recent_improvement = val_loss_avgs[-3] - val_loss_avgs[-1]
        if recent_improvement < 0.01:
            print(f"Note: Validation loss improvement has slowed (last 3 gens: {recent_improvement:.4f})")
            print("      Consider adjusting mutation rate or population diversity.")
    
    print(f"\nPlots saved to: {os.path.abspath(plots_dir)}/")

if __name__ == "__main__":
    main()
