"""
Analysis and Visualization Script for Overcooked-AI Optimization Results

Author: Robert Kasumba (rkasumba@wustl.edu)

This script generates comprehensive analysis and visualizations comparing our optimization
approach against baseline methods. It creates performance plots, statistical analysis,
and summary reports for the Overcooked-AI goal recognition design experiments.

This script is a direct translation from the Jupyter notebook to ensure exact behavior matching.
"""

import sys
sys.path.insert(0, "../")
import torch
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
from torchvision.models import resnet50, resnet18
import argparse
import traceback
import seaborn as sns
import pandas as pd
import csv
import os, re
import ast
import numpy as np
import pickle
from scipy.interpolate import make_interp_spline
from scipy.ndimage import uniform_filter1d

# Constants from notebook
GRID_SIZE = 6
time_out = 18000  # total time across 15 budget evaluations
time_out_per_budget = time_out/15
optimality = "OPTIMAL"
n_lambdas = 17
optimality_folder = "optim_runs" if optimality == "OPTIMAL" else "suboptimal_runs"

cost = 0
base_data_dir = "./baselines/data"  # "./data"
timeout_time = 20

def extract_labels(folder_path, grid_size=6):
    """
    Extract labels from the given folder path.
    """
    # List to hold folder names
    folder_names = []

    # Iterate over all entries in the given folder
    for entry in os.listdir(folder_path):
        # Check if it's a directory
        full_path = os.path.join(folder_path, entry)
        if os.path.isdir(full_path):
            # Add the folder name to the list
            if entry in display_label.keys():
                folder_names.append(entry)

    return folder_names

display_label = {
    "GRAD_valonly": "Our approach",
    "GREEDY_PRED_WCD_CONSTRAINED": "Greedy (predicted wcd)",
    "GREEDY_TRUE_WCD": "Greedy (true wcd)",
    "GREEDY_TRUE_WCD_CONSTRAINED": "Greedy (true wcd)"
}

display_label_colors = {
    "GRAD_valonly": "#66a61e",  # A greenish color
    "GREEDY_PRED_WCD_CONSTRAINED": "#e7298a",  # An orangish color
    "GREEDY_TRUE_WCD": "#7570b3",  # A bluish color
    "GREEDY_TRUE_WCD_CONSTRAINED": "#7570b3",
    "BLOCKING_ONLY_GREEDY_PRED_WCD": "#e7298a",  # A crimson-like pink color
    "BLOCKING_ONLY_test": "#66a61e",  # A grass green color
    "ALL_MODS_EXHUASTIVE": "#1b9e77",  # A golden color
    "ALL_MODS_GREEDY_PRED_WCD": "#e7298a",   # A crimson-like pink color
    "ALL_MODS_GREEDY_TRUE_WCD": "#7570b3",  # A bluish color
    "ALL_MODS_test": "#66a61e"  # A grass green color
}

def read_env_data(file_name):
    """
    Reads environment data from a pickle file.
    """
    with open(file_name, "rb") as f:
        loaded_dataset = pickle.load(f)
        x_data = []
        y_data = []
        for i in range(loaded_dataset.__len__()):
            x_data.append(loaded_dataset[i][0].unsqueeze(0))
            y_data.append(loaded_dataset[i][1].unsqueeze(0))

        x_init_data = torch.cat(x_data).numpy()
        y_init_data = torch.cat(y_data).numpy()
        
    return x_init_data, y_init_data

def read_csv(filename):
    """
    Reads numbers from a CSV file and returns them as a list of lists.

    Args:
    filename (str): The name of the CSV file.

    Returns:
    list of lists: Each sublist contains numbers from a row in the CSV file.
    """
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Convert each string in the row to an integer
            number_row = [ast.literal_eval(item) for item in row]
            data.append(number_row)
    return data 

def moving_average(data, window_size):
    """
    Computes moving average with a given window size.
    """
    return uniform_filter1d(data, size=window_size, mode='reflect')

def plot_summary(df, ylabel, title, show_std_err=True, filename=None, use_given_budget=True, 
                 smoothing_window=0, use_log_scale=False, show_title=False):
    """
    Generate publication-quality plots with error bars for performance comparison.
    """
    fig_size = (20, 16)
    dpi = 300
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi, constrained_layout=True)
    lw = 4
    font_size = 70
    
    for label, row in df.iterrows():
        print(f"Processing {label}...")
        data = pd.DataFrame({
            'budget': row['budget'],
            'mean': row['mean']
        })
            
        # Group by 'budget' and calculate standard error of the mean
        grouped_data = data.groupby('budget')['mean'].agg(['mean', 'sem', "count"]).reset_index()
        
        if use_log_scale:
            grouped_data['sem'] = grouped_data['sem'] / grouped_data['mean']
            grouped_data["mean"] = np.log10(grouped_data["mean"])
            
        # Filter data: only positive budgets and odd budget values
        grouped_data = grouped_data[grouped_data["budget"] > 0]
        grouped_data = grouped_data[grouped_data['budget'] % 2 != 0]
        
        # Apply smoothing if specified
        if smoothing_window:
            grouped_data['mean'] = moving_average(grouped_data['mean'], smoothing_window)
        
        legend_label = f"{display_label[label]}"
        
        # Plot error bars
        plt.errorbar(grouped_data['budget'], grouped_data['mean'], yerr=grouped_data['sem'], fmt='-o',
                     capsize=1, label=legend_label, color=display_label_colors[label], linewidth=lw)
        
        if show_std_err:
            plt.fill_between(grouped_data["budget"], grouped_data['mean'] - grouped_data['sem'],
                             grouped_data['mean'] + grouped_data['sem'], alpha=0.2, color=display_label_colors[label])

    ax.set_xlabel('budget', fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    if show_title:
        ax.set_title(title)
    ax.legend(fontsize=font_size)

    # Extract handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Define the desired order
    desired_order = [1, 2, 0] 

    # Reorder handles and labels
    handles = [handles[i] for i in desired_order]
    labels = [labels[i] for i in desired_order]

    ax.legend(handles, labels, fontsize=font_size)
    
    ax.tick_params(axis='both', which='major', length=font_size/2)
    
    ax.set_xticks(range(0, max(grouped_data['budget'])+5, 5))
    
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xlim([0, 21])
    
    plt.tight_layout()
    
    if filename:
        save_path = f"./plots/{filename}"
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()

def main():
    """
    Main function that replicates the exact behavior from the notebook.
    """
    print("Starting Overcooked-AI Analysis and Visualization...")
    print(f"Grid Size: {GRID_SIZE}, Timeout: {time_out}, Optimality: {optimality}")
    
    # Create plots directory
    os.makedirs("./plots/", exist_ok=True)
    print("Created plots directory: ./plots/")
    
    # Set up experiment parameters
    ratio = "0_0"
    select_idx = 500
    
    # Extract experiment labels - matching notebook logic exactly
    experiment_labels = extract_labels(f"../data/grid{GRID_SIZE}/{optimality_folder}", grid_size=GRID_SIZE) + \
                       extract_labels(f"../baselines/data/grid{GRID_SIZE}/{optimality_folder}/timeout_{time_out}/", grid_size=GRID_SIZE)
    experiment_labels = experiment_labels + ["GRAD_valonly"]
    experiment_labels = ["GREEDY_PRED_WCD_CONSTRAINED", "GREEDY_TRUE_WCD_CONSTRAINED", "GRAD_valonly"]
    constraint = "CONSTRAINED"
    
    # Process experiments and create selections
    selections = [True] * select_idx
    for experiment_label in experiment_labels:
        if "GRAD" in experiment_label:
            base_data_dir = f"../data/grid{GRID_SIZE}/{optimality_folder}/{constraint}/ratio_{ratio}/n_lambdas_{n_lambdas}"
        else:
            base_data_dir = f"../baselines/data/grid{GRID_SIZE}/{optimality_folder}/timeout_{time_out}/{experiment_label}/ratio_{ratio}"
            
        times = np.array(read_csv(f'{base_data_dir}/times_{GRID_SIZE}_{experiment_label}.csv'))
        print(f"Processing {experiment_label}: {times.shape[0]} experiments")
        times = times[0:select_idx]
        
        if ratio == "0_0":
            selections = np.logical_and(selections, (times < time_out_per_budget).all(axis=1))
        else:
            selections = np.logical_and(selections, (times < time_out_per_budget).all(axis=1))
    
    # Initialize DataFrames to store summary data
    columns = ["Experiment Label", "n", 'mean', 'std_err', "budget"]
    time_summary = pd.DataFrame(columns=columns)
    wcd_summary = pd.DataFrame(columns=columns)

    experiment_labels = sorted(experiment_labels, reverse=False)

    time_idx = select_idx
    for experiment_label in experiment_labels:
        if "GRAD" in experiment_label:
            base_data_dir = f"../data/grid{GRID_SIZE}/{optimality_folder}/{constraint}/ratio_{ratio}/n_lambdas_{n_lambdas}"
        else:
            base_data_dir = f"../baselines/data/grid{GRID_SIZE}/{optimality_folder}/timeout_{time_out}/{experiment_label}/ratio_{ratio}"
            if optimality == "SUBOPTIMAL":
                base_data_dir = f"../baselines/data/grid{GRID_SIZE}/{optimality_folder}"
                
        init_file_name = f"{base_data_dir}/initial_envs_{GRID_SIZE}_{experiment_label}.pkl"
        final_file_name = f"{base_data_dir}/final_envs_{GRID_SIZE}_{experiment_label}.pkl"

        times = read_csv(f'{base_data_dir}/times_{GRID_SIZE}_{experiment_label}.csv')
        budgets = read_csv(f'{base_data_dir}/budgets_{GRID_SIZE}_{experiment_label}.csv')
        wcd_change = read_csv(f'{base_data_dir}/wcd_change_{GRID_SIZE}_{experiment_label}.csv')

        budget = np.array(budgets).flatten()
        time_raw = np.array(times)
        times = np.array(times)
        wcd_change = np.array(wcd_change)

        budget = np.array(budgets).flatten()
        times = np.array(times)
        wcd_change = np.array(wcd_change)
            
        print(f"Loading data for {experiment_label}...")
        
        if "GRAD" in experiment_label:
            budgets = np.array(read_csv(f'{base_data_dir}/budgets_{GRID_SIZE}_{experiment_label}.csv'))
            max_budgets = np.array(read_csv(f'{base_data_dir}/max_budgets_{GRID_SIZE}_{experiment_label}.csv') * budgets.shape[0])
            times = times[0:time_idx][selections]
            n = times.shape[0]
            times = times.flatten()
            wcd_change = wcd_change[0:time_idx][selections].flatten()
            budget = max_budgets[0:time_idx][selections].flatten()
            
            print(f"  Loaded {n} experiments with {budget.shape[0]} data points")
            
        else:
            max_budgets = read_csv(f'{base_data_dir}/max_budgets_{GRID_SIZE}_{experiment_label}.csv')
            max_budgets = max_budgets * time_raw.shape[0]  # duplicate

            budgets = np.array(budgets)[0:time_idx][selections]
            max_budgets = np.array(max_budgets)[0:time_idx][selections]  # add one to shift from 0-index to 1- index
            print(f"  Processing max budgets: {max_budgets.shape}")
            times = times[0:time_idx][selections]
            n = times.shape[0]
            
            budget = np.array(max_budgets).flatten()
            times = times.flatten()
            wcd_change = wcd_change[0:time_idx][selections].flatten()

            print(f"  Loaded {n} experiments with {budget.shape[0]} data points")

        time_summary.loc[experiment_label] = [experiment_label, n, times.flatten(), times.flatten(), budget]
        wcd_summary.loc[experiment_label] = [experiment_label, n, wcd_change.flatten(),
                                             wcd_change.flatten(), budget]

    # Store summaries with notebook naming
    time_summary_all_mods_df = time_summary
    wcd_summary_all_mods_df = wcd_summary
    
    print("\nGenerating plots...")
    
    # Create subdirectories for plots
    os.makedirs("./plots/time/", exist_ok=True)
    os.makedirs("./plots/wcd_reduction/", exist_ok=True)
    
    # Plotting Time with log scale
    print("Creating time plot with log scale...")
    plot_summary(time_summary_all_mods_df, 'Mean Log Time (s)', f'', 
                filename=f"time/overcooked_time.pdf", show_std_err=True, use_log_scale=True)
    
    # Plotting WCD Change
    print("Creating WCD reduction plot...")
    plot_summary(wcd_summary_all_mods_df, 'wcd reduction', f'', 
                filename=f"wcd_reduction/overcooked_wcd_reduction.pdf", show_title=True)
    
    print("\nAnalysis complete! All plots have been generated and saved.")

if __name__ == "__main__":
    main()