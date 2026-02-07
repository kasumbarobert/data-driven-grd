"""
Data Preparation Script for Optimization Results Visualization

Author: Robert Kasumba (rkasumba@wustl.edu)

This script processes raw optimization results into analysis-ready formats for visualization
and statistical analysis. It aggregates results across environments and budget levels,
handles timeouts, and creates summary statistics.

WHY THIS IS NEEDED:
- Raw optimization results are stored in individual JSON files per environment
- Analysis requires aggregated data across multiple environments and lambda values
- Visualization tools need properly formatted CSV data with summary statistics
- Timeout handling ensures fair comparison between different approaches

HOW IT WORKS:
1. Reads optimization results from JSON files for each environment
2. Aggregates results across different lambda (cost) values and budget levels
3. Handles experiments that didn't complete within time limits
4. Creates summary statistics (mean, standard error) for visualization
5. Formats data into CSV files suitable for plotting and analysis
6. Supports different ratio constraints (OT:SPD ratios) for constrained optimization

PROCESSING DETAILS:
- Processes results for budget levels from 1 to 40 (step size 2)
- Handles both constrained (ratio-based) and unconstrained optimization results
- Creates separate CSV files for times, WCD changes, and realized budgets
- Manages missing data and timeout scenarios gracefully

USAGE:
    python prepare_optim_results_for_visualization.py

OUTPUT:
    CSV files with aggregated results in data/grid{size}/optim_runs/{constraint}/ratio_{ratio}/n_lambdas_{n}/
"""

import sys
import torch
import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset
from utils import *

# Constants
GRID_SIZE = 6
RATIO = "1_5"  # OT to SPD ratio
max_budgets = list(range(1, 41, 2))

# Utility Functions
def create_folder(folder_path):
    """Create a folder if it doesn't exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")

# Main Experiment Processing Function
def process_experiment(k_values, lambda_values, experiment_label, GRID_SIZE=6, ratio="0_0"):
    """Process optimization results and aggregate them by budget levels."""
    # Initialize storage for aggregated results across all environments
    all_wcd_changes = []
    all_budgets_realized = []
    all_times = []
    data_storage_path = f"data/grid{GRID_SIZE}/optim_runs/CONSTRAINED"
    
    lambda1_values = lambda_values
    lambda2_values = [0]

    # Process each environment's optimization results
    for k in k_values:
        print(f"[figure_5a] Processing environment: {k}", flush=True)
        # Read JSON file containing optimization results for this environment
        file_path = f"{data_storage_path}/langrange_values/env_{k}.json"
        if os.path.exists(file_path):
            with open(file_path, "r") as json_file:
                data = json.load(json_file)

            # Initialize buckets for different budget levels
            budget_buckets_realized = [0 for _ in max_budgets]
            budget_buckets_wcd_change = [0] * len(max_budgets)
            budget_buckets_times = [0] * len(max_budgets)

            # Process each lambda (cost) combination for this environment
            for lambda_pair in data["lambdas"]:
                for i, budget in enumerate(max_budgets):
                    # Extract optimization results
                    num_changes = lambda_pair["num_changes"]
                    wcd_change = lambda_pair["wcd_change"]
                    time_taken = lambda_pair["time_taken"]
                    lambda_value = lambda_pair["lambdas"]
                    
                    # Parse changes into OT (Onion+Tomato) and SPD (Soup+Plate+Dish) components
                    num_changes = [num_changes["O+T"], num_changes["S+P+D"]]
                    
                    # Skip if lambda values don't match our target set
                    if not (lambda_value[0] in lambda1_values and lambda_value[1] in lambda1_values):
                        continue
                    
                    b = budget
                    if ratio != "0_0":
                        # Constrained optimization: check ratio constraints
                        ratio_ot, ratio_spd = map(int, ratio.split('_'))  # Parse ratio (e.g., "1_3" -> 1:3)
                        num_ot = round(b * ratio_ot / (ratio_ot + ratio_spd))
                        num_spd = round(b * ratio_spd / (ratio_ot + ratio_spd))  
                    
                        # Check if changes satisfy ratio constraints
                        if (np.array(num_changes) <= np.array([num_ot, num_spd])).all():
                            # Update best result for this budget level if improvement found
                            if wcd_change >= budget_buckets_wcd_change[i]:
                                budget_buckets_wcd_change[i] = wcd_change
                                budget_buckets_realized[i] = np.sum(num_changes)
                                budget_buckets_times[i] = time_taken
                    else:  # Unconstrained optimization: just check total budget
                        if np.sum(num_changes) <= b:
                            # Update best result for this budget level if improvement found
                            if wcd_change >= budget_buckets_wcd_change[i]:
                                budget_buckets_wcd_change[i] = wcd_change
                                budget_buckets_realized[i] = np.sum(num_changes)
                                budget_buckets_times[i] = time_taken

            # Store results for this environment
            all_wcd_changes.append(budget_buckets_wcd_change)
            all_budgets_realized.append(budget_buckets_realized)
            all_times.append(budget_buckets_times)
            
        else:
            # Handle missing environment files with default values
            print(f"{k} is missing")
            all_wcd_changes.append([0] * len(max_budgets))
            all_budgets_realized.append([0] * len(max_budgets))
            all_times.append([18000] * len(max_budgets))  # Default timeout value
    
    # Save results to CSV files
    n_lambda = len(lambda_values)
    data_storage_path = f"{data_storage_path}/ratio_{ratio}/n_lambdas_{n_lambda}"
    create_folder(data_storage_path)
    create_or_update_list_file(f"{data_storage_path}/times_{GRID_SIZE}_{experiment_label}.csv", all_times)
    create_or_update_list_file(f"{data_storage_path}/wcd_change_{GRID_SIZE}_{experiment_label}.csv", all_wcd_changes)
    create_or_update_list_file(f"{data_storage_path}/budgets_{GRID_SIZE}_{experiment_label}.csv", all_budgets_realized)
    create_or_update_list_file(f"{data_storage_path}/max_budgets_{GRID_SIZE}_{experiment_label}.csv", [list(max_budgets)])

    return all_wcd_changes, all_budgets_realized, all_times

# Function to combine environments
def combine_environments(k_values, lambda_values, experiment_label, GRID_SIZE=6, ratio="1_2", time_out=18000):
    """Process and combine environments with given parameters."""
    # Define lists to store results
    all_wcd_changes = []
    all_budgets_realized = []
    all_times = []
    data_storage_path = f"baselines/data/grid{GRID_SIZE}/optim_runs/timeout_{time_out}/{experiment_label}/ratio_{ratio}"
    
    blocking_rat = 1.05
    unblocking_rat = 1
    lambda1_values = lambda_values
    lambda2_values = lambda_values
    max_budgets = list(range(1, 41, 2))

    # Process data for each environment (k)
    for k in k_values:
        # Read JSON file for current environment
        file_path = f"{data_storage_path}/individual_envs/env_{k}.json"
        if os.path.exists(file_path):
            with open(file_path, "r") as json_file:
                data = json.load(json_file)

            num_changes = [changes["O+T"] + changes["S+P+D"] for changes in data["num_changes"]]
            times = data["times"]
            wcd_changes = data["wcd_changes"]
            
            if len(times) < 20:  # There was a timeout
                target_length = 20
                num_changes.extend([num_changes[-1]] * (target_length - len(num_changes)))
                times.extend([times[-1]] * (target_length - len(times)))
                wcd_changes.extend([wcd_changes[-1]] * (target_length - len(wcd_changes)))
                
            all_budgets_realized.append(num_changes)
            all_wcd_changes.append(wcd_changes)
            all_times.append(times)
        else:
            print(f"{k} is missing")
            all_wcd_changes.append([0] * len(max_budgets))
            all_budgets_realized.append([0] * len(max_budgets))
            all_times.append([18000] * len(max_budgets))
    
    # Save results to CSV files
    n_lambda = len(lambda_values)
    create_or_update_list_file(f"{data_storage_path}/times_{GRID_SIZE}_{experiment_label}.csv", all_times)
    create_or_update_list_file(f"{data_storage_path}/wcd_change_{GRID_SIZE}_{experiment_label}.csv", all_wcd_changes)
    create_or_update_list_file(f"{data_storage_path}/budgets_{GRID_SIZE}_{experiment_label}.csv", all_budgets_realized)
    create_or_update_list_file(f"{data_storage_path}/num_changes_{GRID_SIZE}_{experiment_label}.csv", all_budgets_realized)
    create_or_update_list_file(f"{data_storage_path}/max_budgets_{GRID_SIZE}_{experiment_label}.csv", [max_budgets])
    print(f"Saved results to {data_storage_path}")
    return all_wcd_changes, all_budgets_realized, all_times

# Plotting Function
def plot_results(wcds_by_n, ns):
    """Plot the results from different numbers of lambda pairs."""
    for i, wcds in enumerate(wcds_by_n):
        plt.plot(range(1, 40, 1), wcds, label=f'n={ns[i]}')

    # Add labels and legend
    plt.xlabel('Budget')
    plt.ylabel('WCD Change')
    plt.title('Result from different number of langrange pairs')
    plt.legend()
    plt.show()

# Main function to run the experiments
def main():
    """Main function to handle the processing and plotting of experiments."""
    k_values = range(0, 10000, 20) # sample a few environments i.e every 20 steps
    lambda_values = [0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2, 5, 10]
    experiment_label = "GRAD_valonly"

    all_wcd_changes, all_budgets_realized, all_times = process_experiment(k_values, lambda_values, experiment_label)

    for ratio in ["0_0"]:
        print(f"Combining environments for ratio {ratio}...")
        combine_environments(k_values, lambda_values, "GREEDY_TRUE_WCD_CONSTRAINED", GRID_SIZE=GRID_SIZE, ratio=ratio)
        combine_environments(k_values, lambda_values, "GREEDY_PRED_WCD_CONSTRAINED", GRID_SIZE=GRID_SIZE, ratio=ratio)

if __name__ == "__main__":
    main()
