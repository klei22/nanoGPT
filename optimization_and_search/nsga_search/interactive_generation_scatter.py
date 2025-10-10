#!/usr/bin/env python3
"""
Interactive Generational Scatter Plot with Plotly

Creates an interactive scatter plot showing evolution across generations with:
- Slider to select current generation (highlighted)
- Other generations shown as faded/shaded points
- Separate 2D plots and 3D plot
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from visualization.evolution_history import LogParser
from nsga2 import Population
import json, os


def create_interactive_generational_scatter(file_name_base: str, output_path: str = "htmls/interactive_generational_scatter.html", start_gen: int = 0, end_gen: int = 10):
    
    pop_data = []
    generations = list(range(start_gen, end_gen + 1))

    for gen in generations:
        json_file_name = f"{file_name_base}{gen}.json"
        if not os.path.exists(json_file_name):
            print(f"❌ Checkpoint file not found: {json_file_name}")
            exit(f"❌ Checkpoint file not found: {json_file_name}\nPlease ensure all generation checkpoint files are present.")
                
        population = Population.load_checkpoint(json_file_name, from_pkl=False)

        val_loss_vals = [eva.objs[0] for eva in population.evaluations ]
        energy_vals = [eva.objs[1] for eva in population.evaluations ]
        ttft_vals = [eva.objs[2] for eva in population.evaluations ]

        # Ensure all lists have the same length
        min_len = min(len(val_loss_vals), len(energy_vals), len(ttft_vals))
        
        for i in range(min_len):
            pop_data.append({
                'generation': gen,
                'validation_loss': val_loss_vals[i],
                'energy_per_token': energy_vals[i],
                'ttft': ttft_vals[i],
                'individual_id': i
            })

    df_pop = pd.DataFrame(pop_data)
    df_pop.to_csv("logs/interactive_scatter_population_data.csv", index=False)

    # Create 2D plots
    fig_2d = create_2d_plots(df_pop, generations)
    
    # Create 3D plot
    fig_3d = create_3d_plot(df_pop, generations)
    
    # Save files
    fig_2d.write_html(output_path.replace('.html', '_2d.html'))
    fig_3d.write_html(output_path.replace('.html', '_3d.html'))
    
    print(f"✅ Interactive 2D scatter plot saved to: {output_path.replace('.html', '_2d.html')}")
    print(f"✅ Interactive 3D scatter plot saved to: {output_path.replace('.html', '_3d.html')}")
    
    return


def create_2d_plots(df, generations):
    """Create 2D subplot figure with generation slider"""
    
    # Create subplots: 1 row, 3 columns for the three 2D combinations
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Validation Loss vs Energy/Token', 
                       'Validation Loss vs TTFT', 
                       'Energy/Token vs TTFT'),
        horizontal_spacing=0.1
    )
    
    # Color scheme
    highlight_color = 'red'
    faded_color = 'lightgray'
    
    # Create traces for each generation and each subplot
    for gen in generations:
        gen_data = df[df['generation'] == gen]
        
        if gen_data.empty:
            continue
            
        # Plot 1: Validation Loss vs Energy/Token
        fig.add_trace(
            go.Scatter(
                x=gen_data['energy_per_token'],
                y=gen_data['validation_loss'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=faded_color,
                    opacity=0.3,
                    line=dict(width=1, color='gray')
                ),
                name=f'Gen {gen}',
                text=[f'Gen {gen}, Individual {i}' for i in gen_data['individual_id']],
                hovertemplate='<b>%{text}</b><br>' +
                             'Energy/Token: %{x:.3f}<br>' +
                             'Validation Loss: %{y:.3f}<extra></extra>',
                visible=True,
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Plot 2: Validation Loss vs TTFT
        fig.add_trace(
            go.Scatter(
                x=gen_data['ttft'],
                y=gen_data['validation_loss'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=faded_color,
                    opacity=0.3,
                    line=dict(width=1, color='gray')
                ),
                name=f'Gen {gen}',
                text=[f'Gen {gen}, Individual {i}' for i in gen_data['individual_id']],
                hovertemplate='<b>%{text}</b><br>' +
                             'TTFT: %{x:.3f}<br>' +
                             'Validation Loss: %{y:.3f}<extra></extra>',
                visible=True,
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Plot 3: Energy/Token vs TTFT
        fig.add_trace(
            go.Scatter(
                x=gen_data['energy_per_token'],
                y=gen_data['ttft'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=faded_color,
                    opacity=0.3,
                    line=dict(width=1, color='gray')
                ),
                name=f'Gen {gen}',
                text=[f'Gen {gen}, Individual {i}' for i in gen_data['individual_id']],
                hovertemplate='<b>%{text}</b><br>' +
                             'Energy/Token: %{x:.3f}<br>' +
                             'TTFT: %{y:.3f}<extra></extra>',
                visible=True,
                showlegend=False
            ),
            row=1, col=3
        )
    
    # Add highlighted traces for the first generation (will be controlled by slider)
    for i, gen in enumerate(generations):
        gen_data = df[df['generation'] == gen]
        
        if gen_data.empty:
            continue
        
        # Highlighted traces (initially only first generation visible)
        visible = True if i == 0 else False
        
        # Plot 1 highlighted
        fig.add_trace(
            go.Scatter(
                x=gen_data['energy_per_token'],
                y=gen_data['validation_loss'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=highlight_color,
                    opacity=0.8,
                    line=dict(width=2, color='darkred')
                ),
                name=f'Current: Gen {gen}',
                text=[f'Gen {gen}, Individual {i}' for i in gen_data['individual_id']],
                hovertemplate='<b>%{text}</b><br>' +
                             'Energy/Token: %{x:.3f}<br>' +
                             'Validation Loss: %{y:.3f}<extra></extra>',
                visible=visible,
                showlegend=True if i == 0 else False
            ),
            row=1, col=1
        )
        
        # Plot 2 highlighted
        fig.add_trace(
            go.Scatter(
                x=gen_data['ttft'],
                y=gen_data['validation_loss'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=highlight_color,
                    opacity=0.8,
                    line=dict(width=2, color='darkred')
                ),
                name=f'Current: Gen {gen}',
                text=[f'Gen {gen}, Individual {i}' for i in gen_data['individual_id']],
                hovertemplate='<b>%{text}</b><br>' +
                             'TTFT: %{x:.3f}<br>' +
                             'Validation Loss: %{y:.3f}<extra></extra>',
                visible=visible,
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Plot 3 highlighted
        fig.add_trace(
            go.Scatter(
                x=gen_data['energy_per_token'],
                y=gen_data['ttft'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=highlight_color,
                    opacity=0.8,
                    line=dict(width=2, color='darkred')
                ),
                name=f'Current: Gen {gen}',
                text=[f'Gen {gen}, Individual {i}' for i in gen_data['individual_id']],
                hovertemplate='<b>%{text}</b><br>' +
                             'Energy/Token: %{x:.3f}<br>' +
                             'TTFT: %{y:.3f}<extra></extra>',
                visible=visible,
                showlegend=False
            ),
            row=1, col=3
        )
    
    # Create slider steps
    steps = []
    num_generations = len(generations)
    total_traces = len(fig.data)
    
    for i, gen in enumerate(generations):
        # Calculate which traces should be visible for this generation
        visible_list = [True] * (num_generations * 3)  # Background traces always visible
        
        # Add visibility for highlighted traces
        for j in range(num_generations):
            visible_list.extend([j == i, j == i, j == i])  # Highlight current generation
        
        step = dict(
            method="update",
            args=[{"visible": visible_list}],
            label=f"Gen {gen}"
        )
        steps.append(step)
    
    # Add slider
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Current Generation: "},
        pad={"t": 50},
        steps=steps
    )]
    
    fig.update_layout(
        sliders=sliders,
        title=dict(
            text="Interactive Generational Evolution - 2D Views<br><sub>Use slider to highlight different generations</sub>",
            x=0.5,
            font=dict(size=16)
        ),
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Energy per Token", row=1, col=1)
    fig.update_yaxes(title_text="Validation Loss", row=1, col=1)
    
    fig.update_xaxes(title_text="TTFT", row=1, col=2)
    fig.update_yaxes(title_text="Validation Loss", row=1, col=2)
    
    fig.update_xaxes(title_text="Energy per Token", row=1, col=3)
    fig.update_yaxes(title_text="TTFT", row=1, col=3)
    
    return fig


def create_3d_plot(df, generations):
    """Create 3D scatter plot with generation slider"""
    
    fig = go.Figure()
    
    # Color scheme
    highlight_color = 'red'
    faded_color = 'lightgray'
    
    # Add background traces for all generations (always visible, faded)
    for gen in generations:
        gen_data = df[df['generation'] == gen]
        
        if gen_data.empty:
            continue
            
        fig.add_trace(
            go.Scatter3d(
                x=gen_data['energy_per_token'],
                y=gen_data['ttft'],
                z=gen_data['validation_loss'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=faded_color,
                    opacity=0.3,
                    line=dict(width=1, color='gray')
                ),
                name=f'Gen {gen} (background)',
                text=[f'Gen {gen}, Individual {i}' for i in gen_data['individual_id']],
                hovertemplate='<b>%{text}</b><br>' +
                             'Energy/Token: %{x:.3f}<br>' +
                             'TTFT: %{y:.3f}<br>' +
                             'Validation Loss: %{z:.3f}<extra></extra>',
                visible=True,
                showlegend=False
            )
        )
    
    # Add highlighted traces for each generation (controlled by slider)
    for i, gen in enumerate(generations):
        gen_data = df[df['generation'] == gen]
        
        if gen_data.empty:
            continue
        
        visible = True if i == 0 else False
        
        fig.add_trace(
            go.Scatter3d(
                x=gen_data['energy_per_token'],
                y=gen_data['ttft'],
                z=gen_data['validation_loss'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=highlight_color,
                    opacity=0.8,
                    line=dict(width=2, color='darkred')
                ),
                name=f'Current: Gen {gen}',
                text=[f'Gen {gen}, Individual {i}' for i in gen_data['individual_id']],
                hovertemplate='<b>%{text}</b><br>' +
                             'Energy/Token: %{x:.3f}<br>' +
                             'TTFT: %{y:.3f}<br>' +
                             'Validation Loss: %{z:.3f}<extra></extra>',
                visible=visible,
                showlegend=True if i == 0 else False
            )
        )
    
    # Create slider steps
    steps = []
    num_generations = len(generations)
    
    for i, gen in enumerate(generations):
        # Background traces always visible, highlight only current generation
        visible_list = [True] * num_generations  # Background traces
        visible_list.extend([j == i for j in range(num_generations)])  # Highlighted traces
        
        step = dict(
            method="update",
            args=[{"visible": visible_list}],
            label=f"Gen {gen}"
        )
        steps.append(step)
    
    # Add slider
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Current Generation: "},
        pad={"t": 50},
        steps=steps
    )]
    
    fig.update_layout(
        sliders=sliders,
        title=dict(
            text="Interactive Generational Evolution - 3D View<br><sub>Use slider to highlight different generations</sub>",
            x=0.5,
            font=dict(size=16)
        ),
        scene=dict(
            xaxis_title='Energy per Token',
            yaxis_title='TTFT',
            zaxis_title='Validation Loss',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def main():
    """Main function to create the interactive plots"""
    import os
    
    # take arguments from command line
    import argparse
    
    parser = argparse.ArgumentParser(description="Create Interactive Generational Scatter Plots")
    parser.add_argument("--ckpt_base", type=str, default="ckpts/infi_attn_exp_2/1009_0627_ckpt_gen", help="Path to the evolution log file")
    parser.add_argument("--start_gen", type=int, default=1, help="Starting generation index")
    parser.add_argument("--end_gen", type=int, default=50, help="Ending generation index")
    parser.add_argument("--output", type=str, default="htmls/interactive_generational_scatter.html", help="Output HTML file path")
    args = parser.parse_args()
    
    file_name_base = args.ckpt_base
    start_gen = args.start_gen
    end_gen = args.end_gen
    output_path = args.output

    create_interactive_generational_scatter(file_name_base, output_path, start_gen, end_gen)

    print("\n✅ Interactive plots created!")
    print("📁 Files created:")
    print("   interactive_generational_scatter_2d.html - 2D scatter plots with slider")
    print("   interactive_generational_scatter_3d.html - 3D scatter plot with slider")
    print("\n🌐 Open the HTML files in your browser to interact with the plots!")


if __name__ == "__main__":
    main()
    
# python interactive_generation_scatter.py --ckpt_base="ckpts/infi_attn_exp_2/1009_0627_ckpt_gen" --start_gen=1 --end_gen=30
