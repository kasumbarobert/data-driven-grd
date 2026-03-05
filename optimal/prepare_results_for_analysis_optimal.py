import sys
import os
import torch
from torch.utils.data import Dataset
from utils import *
import random
import matplotlib.pyplot as plt
from torchvision.models import resnet50, resnet18
import argparse
import traceback
import seaborn as sns
import numpy as np
import json

# GRID_SIZE = 13
LARGE_TIME_OUT = 9000
# DATA_STORAGE_BASE_PATH = "data/grid{}/".format(GRID_SIZE)

# Load initial WCD data
def load_initial_wcd(file_path):
    with open(file_path, "r") as json_file:
        return json.load(json_file)


def create_folder(folder_path):
    """Create a folder if it does not exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")

def save_results(data_storage_path, experiment_label, grid_size, all_times, all_wcd_changes, all_budgets_realized, max_budgets):
    """Save results to CSV files."""
    # n_lambda = len(lambda_values)
    full_path = data_storage_path
    create_folder(full_path)
    create_or_update_list_file(f"{full_path}/times_{grid_size}_{experiment_label}.csv", all_times)
    create_or_update_list_file(f"{full_path}/wcd_change_{grid_size}_{experiment_label}.csv", all_wcd_changes)
    create_or_update_list_file(f"{full_path}/num_changes_{grid_size}_{experiment_label}.csv", all_budgets_realized)
    create_or_update_list_file(f"{full_path}/budgets_{grid_size}_{experiment_label}.csv", [max_budgets])

def process_experiment(k_values, lambda_values, blocking_rat, unblocking_rat, experiment_label, initial_true_wcd,max_budgets, grid_size=13, 
                    input_data_dir = f"data/grid6/ALL_MODS_test/", output_dir = f"data/grid6"):
    """Process the experiment with given parameters."""
    data_storage_path = f"{output_dir}/{experiment_label}/ratio_{blocking_rat}_{unblocking_rat}"
    if experiment_label == "BLOCKING_ONLY_test":
        blocking_rat = 1
        unblocking_rat = 0
        data_storage_path = f"{output_dir}/{experiment_label}"
    elif experiment_label == "BOTH_UNIFORM_test":
        data_storage_path = f"{output_dir}/{experiment_label}"

    data_storage_path = f"{data_storage_path}/n_lambdas_{len(lambda_values)}/"
    
    lambda1_values = lambda_values
    lambda2_values = lambda_values

    all_wcd_changes, all_budgets_realized, all_times = [], [], []

    

    for k in k_values:
        budget_buckets_realized = [[0, 0] for _ in max_budgets]
        budget_buckets_wcd_change = [0] * len(max_budgets)
        budget_buckets_times = [0] * len(max_budgets)
        ignore_times = [LARGE_TIME_OUT] * len(max_budgets) # to be used for missing envs or initial env is zero
        try:
            file_path = f"{input_data_dir}/langrange_values/env_{k}.json"
            with open(file_path, "r") as json_file:
                data = json.load(json_file)

            for lambda_pair in data["lambda_pairs"]:
                for i, budget in enumerate(max_budgets):
                    max_changes_dist = np.round([
                        (blocking_rat * budget) / (unblocking_rat + blocking_rat),
                        (unblocking_rat * budget) / (unblocking_rat + blocking_rat)
                    ]).tolist()

                    num_changes, wcd_change, time_taken, lambdas = (
                        lambda_pair["num_changes"],
                        lambda_pair["wcd_change"],
                        lambda_pair["time_taken"],
                        lambda_pair["lambdas"],
                    )

                    if lambdas[0] in lambda1_values and lambdas[1] in lambda2_values:
                        if experiment_label=="BOTH_UNIFORM_test": # total is what matters not individual budgets
                            if np.sum(num_changes) <= budget:
                                if wcd_change > budget_buckets_wcd_change[i]:  # found a better value
                                    budget_buckets_wcd_change[i] = wcd_change
                                    budget_buckets_realized[i] = num_changes
                                    budget_buckets_times[i] = time_taken
                        else:
                            if (np.array(num_changes) <= np.array(max_changes_dist)).all():
                                if wcd_change >= budget_buckets_wcd_change[i]:
                                    budget_buckets_wcd_change[i] = wcd_change
                                    budget_buckets_realized[i] = num_changes
                                    budget_buckets_times[i] = time_taken
        except:
            budget_buckets_times = ignore_times 
        # Handle cases where initial WCD is zero
        if str(k) in initial_true_wcd and initial_true_wcd[str(k)] == 0:
            budget_buckets_times = ignore_times

        all_wcd_changes.append(budget_buckets_wcd_change)
        all_budgets_realized.append(budget_buckets_realized)
        all_times.append(budget_buckets_times)

    save_results(data_storage_path, experiment_label, grid_size, all_times, all_wcd_changes, all_budgets_realized, max_budgets)
    return all_wcd_changes, all_budgets_realized, all_times


def process_experiment_baselines(k_values, blocking_rat, unblocking_rat, experiment_label, initial_true_wcd,  max_budgets, grid_size=13, input_data_dir = f"data/grid6/ALL_MODS_test/", output_dir = f"data/grid6"):
    """Process the experiment with given parameters."""
    
    if "ALL_MODS" in experiment_label:
        data_storage_path = f"{output_dir}{experiment_label}/ratio_{blocking_rat}_{unblocking_rat}/"
        input_data_dir = f"{input_data_dir}/{experiment_label}/ratio_{blocking_rat}_{unblocking_rat}/"
    else:
        data_storage_path = f"{output_dir}/{experiment_label}/"
        input_data_dir = f"{input_data_dir}/{experiment_label}/"

    all_wcd_changes, all_budgets_realized, all_times = [], [], []
    # if "ALL_MODS" in experiment_label:
    #     import pdb; pdb.set_trace()
    for k in k_values:
        budget_buckets_realized = [[0, 0] for _ in max_budgets]
        budget_buckets_wcd_change = [0] * len(max_budgets)
        budget_buckets_times = [0] * len(max_budgets)
        try:
            file_path = f"{input_data_dir}/env_modifications/envs_{k}.json"
            with open(file_path, "r") as json_file:
                data = json.load(json_file)
            budget_buckets_realized = data["num_changes"]
            budget_buckets_wcd_change = data["wcd_change"]
            budget_buckets_times = data["times"]
            init_wcd = data["init_wcd"]
            # max_budgets = data["given_budget"]

            # Handle cases where initial WCD is zero - IGNORE these environments
            if init_wcd==0:
                budget_buckets_times = ignore_times
        
        except:
            ignore_times = [LARGE_TIME_OUT] * len(max_budgets) # to be used for missing envs or initial env is zero

        all_wcd_changes.append(budget_buckets_wcd_change)
        all_budgets_realized.append(budget_buckets_realized)
        all_times.append(budget_buckets_times)

    save_results(data_storage_path, experiment_label, grid_size, all_times, all_wcd_changes, all_budgets_realized, max_budgets)
    return all_wcd_changes, all_budgets_realized, all_times

def run_single_analysis(args, k_values, lambda_values, max_budgets, initial_true_wcd, noise_level=None):
    """Run analysis for a single noise level or the default case."""
    
    # Determine output and input directories based on noise level
    if noise_level is not None:
        output_dir = f"summary_data/grid{args.grid_size}/ml-our-approach/{args.wcd_pred_model_id}_sensitivity_analysis_{noise_level}_noise/"
        input_data_dir = f"wcd_optim_results/ml-our-approach/grid{args.grid_size}/{args.wcd_pred_model_id}_sensitivity_analysis_{noise_level}_noise/ALL_MODS_test/"
    else:
        output_dir = f"summary_data/grid{args.grid_size}/ml-our-approach/{args.wcd_pred_model_id}/"
        if args.wcd_pred_model_id == "aaai25_submission":
            input_data_dir = f"data/grid{args.grid_size}/ALL_MODS_test/"
        else:
            input_data_dir = f"wcd_optim_results/ml-our-approach/grid{args.grid_size}/{args.wcd_pred_model_id}/ALL_MODS_test/"
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_data_dir):
        print(f"Warning: No input directory {input_data_dir} found... skipping.")
        return
    
    # Set blocking/unblocking ratios based on grid size
    if args.grid_size == 13:
        blocking_rat = 5
        unblocking_rat = 1
    else:
        blocking_rat = 3
        unblocking_rat = 1
    
    print(f"Processing analysis for noise_level: {noise_level if noise_level is not None else 'default'}")
    
    # Run main experiments
    all_wcd_changes, all_budgets_realized, all_times = process_experiment(
        k_values, lambda_values, blocking_rat=blocking_rat, unblocking_rat=unblocking_rat, 
        experiment_label="ALL_MODS_test", initial_true_wcd=initial_true_wcd, 
        grid_size=args.grid_size, input_data_dir=input_data_dir, output_dir=output_dir,
        max_budgets=max_budgets
    )
    
    all_wcd_changes, all_budgets_realized, all_times = process_experiment(
        k_values, lambda_values, blocking_rat=1, unblocking_rat=0, 
        experiment_label="BLOCKING_ONLY_test", initial_true_wcd=initial_true_wcd, 
        grid_size=args.grid_size, input_data_dir=input_data_dir, output_dir=output_dir, 
        max_budgets=max_budgets
    )
    
    all_wcd_changes, all_budgets_realized, all_times = process_experiment(
        k_values, lambda_values, blocking_rat=1, unblocking_rat=1, 
        experiment_label="BOTH_UNIFORM_test", initial_true_wcd=initial_true_wcd, 
        grid_size=args.grid_size, input_data_dir=input_data_dir, output_dir=output_dir, 
        max_budgets=max_budgets
    )
    
    # Run baseline experiments
    baseline_exp_labels = ["BLOCKING_ONLY_EXHAUSTIVE", "BLOCKING_ONLY_PRUNE_REDUCE", "BLOCKING_ONLY_GREEDY_TRUE_WCD", 
                          "BLOCKING_ONLY_GREEDY_PRED_WCD", "ALL_MODS_EXHAUSTIVE", "ALL_MODS_GREEDY_TRUE_WCD", 
                          "ALL_MODS_GREEDY_PRED_WCD", "BOTH_UNIFORM_EXHAUSTIVE", "BOTH_UNIFORM_GREEDY_PRED_WCD", 
                          "BOTH_UNIFORM_GREEDY_TRUE_WCD"]
    
    if args.sensitivity_analysis: # no need to run baselines for sensitivity analysis
        return
    
    for experiment_label in baseline_exp_labels:
        if "PRED_WCD" in experiment_label:
            baseline_output_dir = f"summary_data/grid{args.grid_size}/ml-greedy/{args.wcd_pred_model_id}/timeout_600/"
            baseline_input_data_dir = f"wcd_optim_results/ml-greedy/grid{args.grid_size}/timeout_600/{args.wcd_pred_model_id}/"
        else:
            baseline_output_dir = f"summary_data/grid{args.grid_size}/greedy/timeout_600/"
            baseline_input_data_dir = f"wcd_optim_results/greedy/grid{args.grid_size}/timeout_600/"
        
        os.makedirs(baseline_output_dir, exist_ok=True)
        
        process_experiment_baselines(k_values, blocking_rat, unblocking_rat, experiment_label=experiment_label, 
                                   initial_true_wcd=initial_true_wcd, max_budgets=max_budgets, 
                                   grid_size=args.grid_size, input_data_dir=baseline_input_data_dir, 
                                   output_dir=baseline_output_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Simulate data")
    
    parser.add_argument(
        "--grid_size",
        type=int,  # Ensure that the input is expected to be a int
        default=13,  # Set the default value to 1
        help="Maximum grid size.",
    )
    parser.add_argument(
        "--wcd_pred_model_id",
        type=str,  # Ensure that the input is expected to be a string
        default="aaai25_submission",  # Set the default value to 1
        help="model id used for optimization - determines the files to use",
    )
    parser.add_argument(
        "--sensitivity_analysis",
        default=False,  # Set the default value to 1
        action="store_true", # store the value as True if the argument is present
        help="Whether to use sensitivity analysis",
    )

    args = parser.parse_args()

    print(f"Running with the following configurations {args}")
    
    # Example usage
    k_values = range(0, 7014, 14) if args.grid_size == 13 else range(0, 14252, 28) # if grid_size is 6, for aaai 13 -> 3136
    lambda_values = [0, 0.001, 0.002, 0.005, 0.007, 0.01, 0.02, 0.05, 0.07, 0.1, 0.2, 0.5, 0.7, 1.0, 2, 5, 7]
    max_budgets = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41]

    initial_true_wcd = load_initial_wcd(f"data/grid{args.grid_size}/initial_true_wcd_by_id.json")

    if args.sensitivity_analysis:
        # Define noise levels for sensitivity analysis
        noise_levels = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        
        for noise_level in noise_levels:
            print(f"\n=== Running sensitivity analysis for noise_level: {noise_level} ===")
            run_single_analysis(args, k_values, lambda_values, max_budgets, initial_true_wcd, noise_level)
    else:
        # Run single analysis without noise
        print("\n=== Running standard analysis ===")
        run_single_analysis(args, k_values, lambda_values, max_budgets, initial_true_wcd)