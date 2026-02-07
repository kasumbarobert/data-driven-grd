"""Converted from construct_budget_wcd_change -- suboptimal.ipynb.
Generated automatically by tools/notebook_to_script.py.
"""

import csv
import os
import sys
import torch
from torch.utils.data import Dataset
from utils_suboptimal import *
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.models import resnet50, resnet18
import argparse
import traceback
import seaborn as sns
import numpy as np
import json

# %% [code] cell 0
# sys.path.insert(0, "./")
# sys.path.insert(0, "../../")

# %% [code] cell 1
K = 8
GRID_SIZE = 6
interval = 53 if GRID_SIZE ==10 else 1 # for grid size 6
n = 600 if GRID_SIZE ==10 else 2200
TIMEOUT = 600
IGNORE_ENV_TIMOUT_VALUE = 2*TIMEOUT

# %% [code] cell 2
data_storage_path =f"data/grid{GRID_SIZE}/K{K}/"
with open(f"{data_storage_path}initial_true_wcd_by_id.json", "r") as json_file:
    initial_true_wcd = json.load(json_file)
len(initial_true_wcd)

# %% [code] cell 3
def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")

# %% [code] cell 4
def process_experiment(k_values, lambda_values,experiment_label, GRID_SIZE=10):
    """ Process the experiment with given parameters. """
    # Define lists to store results
    all_wcd_changes = []
    all_budgets_realized = []
    all_times = []
    data_storage_path = f"data/grid{GRID_SIZE}/K{K}/{experiment_label}"

    blocking_rat = 1.05
    unblocking_rat = 1
    data_storage_path = f"data/grid{GRID_SIZE}/K{K}/{experiment_label}"


    lambda1_values = lambda_values
    lambda2_values = lambda_values #[0]
    # Process data for each k
    for k in k_values:
        # Read JSON file for current k
        file_path = f"data/grid{GRID_SIZE}/K{K}/ALL_MODS_test/langrange_values/env_{k}.json"
        budget_buckets_realized = [[0,0] for _ in max_budgets]
        budget_buckets_wcd_change = [0] * len(max_budgets)
        budget_buckets_times = [0] * len(max_budgets)

        if  os.path.exists(file_path):
            with open(file_path, "r") as json_file:
                data = json.load(json_file)
                # print(json_file)
            if len(data["lambda_pairs"])<256:
                print(k,"has",len(data["lambda_pairs"]))
            for lambda_pair in data["lambda_pairs"]:
                for i, budget in enumerate(max_budgets):
                    num_changes = lambda_pair["num_changes"]
                    wcd_change = lambda_pair["wcd_change"]
                    time_taken = lambda_pair["time_taken"]
                    lambdas = lambda_pair["lambdas"]

                    if not (lambdas[0] in lambda1_values and lambdas[1] in lambda2_values):
                        continue

                    if np.sum(num_changes) <= budget:
                        if wcd_change > budget_buckets_wcd_change[i]:  # found a better value
                            budget_buckets_wcd_change[i] = wcd_change
                            budget_buckets_realized[i] = num_changes
                            budget_buckets_times[i] = time_taken
            # if len(data["lambda_pairs"])<256:
            #     budget_buckets_times = [TIMEOUT] * len(max_budgets)
        else:
            # print(f"env_{k}.json missing")
            # continue
            budget_buckets_times = [TIMEOUT] * len(max_budgets)

        ignore_times= [IGNORE_ENV_TIMOUT_VALUE] * len(max_budgets) #IGNORE THESE

        if str(k) in initial_true_wcd.keys():#Init true WCD likely 0
            if initial_true_wcd[str(k)]==0:
                budget_buckets_times = ignore_times
        else:
            budget_buckets_times = ignore_times

        # Append results for current k to the lists of all results
        all_wcd_changes.append(budget_buckets_wcd_change)
        all_budgets_realized.append(budget_buckets_realized)
        all_times.append(budget_buckets_times)

    # Save results to CSV files
    n_lambda = len(lambda_values)
    data_storage_path = f"{data_storage_path}/n_lambdas_{n_lambda}"
    create_folder(data_storage_path)
    create_or_update_list_file(f"{data_storage_path}/times_{GRID_SIZE}_{experiment_label}.csv", all_times)
    create_or_update_list_file(f"{data_storage_path}/wcd_change_{GRID_SIZE}_{experiment_label}.csv", all_wcd_changes)
    create_or_update_list_file(f"{data_storage_path}/budgets_{GRID_SIZE}_{experiment_label}.csv", all_budgets_realized)
    create_or_update_list_file(f"{data_storage_path}/max_budgets_{GRID_SIZE}_{experiment_label}.csv",[max_budgets])
    return all_wcd_changes, all_budgets_realized, all_times

# Example usage
k_values = range(0, interval*n, interval)
lambda_values = [0,0.0001, 0.0002, 0.0005,0.001, 0.002, 0.005,0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2, 5, 10]
max_budgets = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35][0:10]

experiment_label ="BOTH_UNIFORM_test"

# %% [code] cell 5
data_storage_path = f"data/grid{GRID_SIZE}/K{K}"
init_true_wcd_csv = []
for k in k_values:
    init_true_wcd_csv.append(initial_true_wcd[str(k)])
create_or_update_list_file(f"{data_storage_path}/selected_env_init_true_wcd.csv",[init_true_wcd_csv])

# %% [code] cell 6
temp_values = np.array(lambda_values)
wcds_by_n = []
ns = []
i = 0
while True:
    all_wcd_changes, all_budgets_realized, all_times = process_experiment(k_values, temp_values, experiment_label,GRID_SIZE = GRID_SIZE)
    wcds_by_n.append(np.mean(all_wcd_changes, axis=0))
    ns.append(temp_values.shape[0]**2)
    temp_values *= 10
    temp_values = temp_values[temp_values <= 10]


    i += 1
    if i >= 1:
        break

# %% [code] cell 7
# # Plotting the curves
# for i, wcds in enumerate(wcds_by_n):
#     print(i)
#     plt.plot(range(1,36,2),wcds, label=f'n={ns[i]}')

# # Add labels and legend
# plt.xlabel('Budget')
# plt.ylabel('WCD Change')
# plt.title('Result from different number of langrange pairs')
# plt.legend()

# # Show plot
# plt.show()

# %% [code] cell 8
def process_experiment(k_values,experiment_label, GRID_SIZE=10, timeout = 1800):
    """ Process the experiment with given parameters. """
    # Define lists to store results
    all_wcd_changes = []
    all_budgets_realized = []
    all_times = []
    data_storage_path = f"baselines/data/grid{GRID_SIZE}/K{K}//timeout_{timeout}/{experiment_label}"

    data_storage_path = f"baselines/data/grid{GRID_SIZE}/K{K}//timeout_{timeout}/{experiment_label}"

    max_budgets = 19
    # Process data for each k
    for k in k_values:
        # Read JSON file for current k
        file_path = f"baselines/data/grid{GRID_SIZE}/K{K}/timeout_{timeout}/{experiment_label}/individual_envs/env_{k}.json"
        budget_buckets_realized = [[0,0] for _ in range(max_budgets)]
        budget_buckets_wcd_change = [0] * max_budgets
        budget_buckets_times = [0] * max_budgets

        if  os.path.exists(file_path):
            with open(file_path, "r") as json_file:
                data = json.load(json_file)


            budget_buckets_realized = data["num_changes"]
            budget_buckets_wcd_change = data["wcd_changes"]
            budget_buckets_times = data["times"]
            max_budgets = data["max_budgets"][0]

        else:
            # print(f"env_{k}.json missing")
            # continue
            budget_buckets_times = [TIMEOUT] * max_budgets

        # if budget_buckets_times[-1]<TIMEOUT and budget_buckets_wcd_change[-1]==0:#Init true WCD likely 0
        #     # budget_buckets_times = [TIMEOUT] * len(max_budgets)
        #     budget_buckets_times = [TIMEOUT] * max_budgets

        ignore_times= [IGNORE_ENV_TIMOUT_VALUE] * max_budgets #IGNORE THESE
        if str(k) in initial_true_wcd.keys():#Init true WCD likely 0
            if initial_true_wcd[str(k)]==0:
                budget_buckets_times = ignore_times
        else:
            budget_buckets_times = ignore_times

        # Append results for current k to the lists of all results
        all_wcd_changes.append(budget_buckets_wcd_change)
        all_budgets_realized.append(budget_buckets_realized)
        all_times.append(budget_buckets_times)

    # Save results to CSV files

    create_folder(data_storage_path)
    create_or_update_list_file(f"{data_storage_path}/times_{GRID_SIZE}_{experiment_label}.csv", all_times)
    create_or_update_list_file(f"{data_storage_path}/wcd_change_{GRID_SIZE}_{experiment_label}.csv", all_wcd_changes)
    create_or_update_list_file(f"{data_storage_path}/num_changes_{GRID_SIZE}_{experiment_label}.csv", all_budgets_realized)
    create_or_update_list_file(f"{data_storage_path}/budgets_{GRID_SIZE}_{experiment_label}.csv", all_budgets_realized)
    create_or_update_list_file(f"{data_storage_path}/max_budgets_{GRID_SIZE}_{experiment_label}.csv",[[k for k in range(max_budgets)]])
    return all_wcd_changes, all_budgets_realized, all_times

# Example usage
k_values = range(0, interval*n, interval)

# experiment_label ="BOTH_UNIFORM_GREEDY_TRUE_WCD"

# %% [code] cell 9
process_experiment(k_values, "BOTH_UNIFORM_GREEDY_TRUE_WCD",GRID_SIZE = GRID_SIZE, timeout = TIMEOUT)
process_experiment(k_values, "BOTH_UNIFORM_GREEDY_PRED_WCD",GRID_SIZE = GRID_SIZE, timeout = TIMEOUT)
print("done")
