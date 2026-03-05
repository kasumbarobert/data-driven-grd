"""Converted from construct_budget_ratios--oc.ipynb.
Generated automatically by tools/notebook_to_script.py.
"""

import os
import sys
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

# %% [code] cell 0
# sys.path.insert(0, "./")
# sys.path.insert(0, "../../")

# %% [code] cell 1
452/2

# %% [code] cell 2
GRID_SIZE=6
assumed_behavior =  "HUMAN" #"OPTIMAL" #"HUMAN"

if assumed_behavior == "OPTIMAL":
    interval =  226 if GRID_SIZE ==6 else 590
    max_id = interval*150 if GRID_SIZE ==6 else interval*11
else:
    interval =  226 if GRID_SIZE ==6 else 590
    max_id = interval*150 if GRID_SIZE ==6 else interval*11

# %% [code] cell 3
def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    # else:
    #     print(f"Folder '{folder_path}' already exists.")

# %% [code] cell 4
def process_experiment(k_values, lambda_values, blocking_rat, unblocking_rat, experiment_label, GRID_SIZE=10):
    """Process the experiment with given parameters."""
    # Define lists to store results
    all_wcd_changes = []
    all_budgets_realized = []
    all_times = []
    data_storage_path = f"data/grid{GRID_SIZE}/{assumed_behavior}/ratio_{blocking_rat}_{unblocking_rat}"
    if experiment_label =="BLOCKING_ONLY_test":
        blocking_rat = 1
        unblocking_rat = 0
        data_storage_path = f"data/grid{GRID_SIZE}/{experiment_label}"

    lambda1_values = lambda_values
    lambda2_values = lambda_values
    # Process data for each k
    for k in k_values:
        # Read JSON file for current k

        file_path = f"data/grid{GRID_SIZE}/{assumed_behavior}/langrange_values/env_{k}.json"
        budget_buckets_realized = [[0,0] for _ in max_budgets]
        budget_buckets_wcd_change = [0] * len(max_budgets)
        budget_buckets_times = [0] * len(max_budgets)
        budget_best_lambdas = [[0,0]] * len(max_budgets)

        if  os.path.exists(file_path):
            with open(file_path, "r") as json_file:
                data = json.load(json_file)
            for lambda_pair in data["lambda_pairs"]:
                # print(lambda_pair)
                for i, budget in enumerate(max_budgets):
                    max_changes_dist = np.round([(blocking_rat * budget) / (unblocking_rat + blocking_rat),
                                                 (unblocking_rat * budget) / (unblocking_rat + blocking_rat)]).tolist()

                    num_changes = lambda_pair["num_changes"]
                    wcd_change = lambda_pair["wcd_change"]
                    time_taken = lambda_pair["time_taken"]
                    lambdas = lambda_pair["lambdas"]

                    if not (lambdas[0] in lambda1_values and lambdas[1] in lambda2_values):
                        continue
                    # print(lambda_pair["lambdas"],np.array(num_changes) <= np.array(max_changes_dist),np.array(num_changes) , np.array(max_changes_dist))
                    if (np.array(num_changes) <= np.array(max_changes_dist)).all():
                        if wcd_change >= budget_buckets_wcd_change[i]:  # found a better value
                            budget_buckets_wcd_change[i] = wcd_change
                            budget_buckets_realized[i] = num_changes
                            budget_buckets_times[i] = time_taken
                            budget_best_lambdas[i]= lambdas
        else:
            print(f"env_{k}.json missing")
            # continue
            budget_buckets_times = [700] * len(max_budgets)


        # Append results for current k to the lists of all results
        all_wcd_changes.append(budget_buckets_wcd_change)
        all_budgets_realized.append(budget_buckets_realized)
        all_times.append(budget_buckets_times)

    # Save results to CSV files
    n_lambda = len(lambda_values)
    data_storage_path = f"{data_storage_path}/n_lambdas_{n_lambda**2}"
    create_folder(data_storage_path)
    create_or_update_list_file(f"{data_storage_path}/times_{GRID_SIZE}_{experiment_label}.csv", all_times)
    create_or_update_list_file(f"{data_storage_path}/wcd_change_{GRID_SIZE}_{experiment_label}.csv", all_wcd_changes)
    create_or_update_list_file(f"{data_storage_path}/budgets_{GRID_SIZE}_{experiment_label}.csv", all_budgets_realized)
    create_or_update_list_file(f"{data_storage_path}/max_budgets_{GRID_SIZE}_{experiment_label}.csv",[max_budgets])
    create_or_update_list_file(f"{data_storage_path}/best_lambdas_{GRID_SIZE}_{experiment_label}.csv",budget_best_lambdas)
    return all_wcd_changes, all_budgets_realized, all_times

# Example usage

k_values = range(0, max_id, interval)
lambda_values = [0,0.001,0.002,0.005,0.007,0.01,0.02,0.05,0.07,0.1,0.2,0.5,0.7,1.0,2,5,7]
max_budgets = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23,25,27,29,31,33,35,37,39,41]
blocking_rat = 3
unblocking_rat = 1
experiment_label =f"{assumed_behavior}_ALL_MODS_test" #"ALL_MODS_test"#"BLOCKING_ONLY_test"


all_wcd_changes, all_budgets_realized, all_times = process_experiment(k_values, lambda_values, blocking_rat, unblocking_rat, experiment_label,GRID_SIZE=GRID_SIZE)

# %% [code] cell 5
temp_values = np.array(lambda_values)
wcds_by_n = []
ns = []
i = 0
while True:
    all_wcd_changes, all_budgets_realized, all_times = process_experiment(k_values, temp_values, blocking_rat, unblocking_rat, experiment_label,GRID_SIZE=GRID_SIZE)
    wcds_by_n.append(np.mean(all_wcd_changes, axis=0))
    ns.append(temp_values.shape[0]**2)
    print(temp_values)
    temp_values *= 10
    temp_values = temp_values[temp_values <= 10]


    i += 1
    if i >= 5:
        break

# %% [code] cell 6
# Plotting the curves
for i, wcds in enumerate(wcds_by_n):
    print(i)
    plt.plot(max_budgets,wcds, label=f'n={ns[i]}')

# Add labels and legend
plt.xlabel('Budget')
plt.ylabel('WCD Change')
plt.title('Result from different number of langrange pairs')
plt.legend()

# Show plot
plt.show()

# %% [code] cell 8
def process_experiment(k_values, lambda_values,experiment_label, GRID_SIZE=10):
    """Process the experiment with given parameters."""
    # Define lists to store results
    all_wcd_changes = []
    all_budgets_realized = []
    all_times = []
    all_best_lambdas =[]
    data_storage_path = f"data/grid{GRID_SIZE}/{assumed_behavior}/{experiment_label}"

    blocking_rat = 1.05
    unblocking_rat = 1
    data_storage_path = f"data/grid{GRID_SIZE}/{assumed_behavior}/{experiment_label}"

    lambda1_values = lambda_values
    lambda2_values = lambda_values
    # Process data for each k
    for k in k_values:
        # Read JSON file for current k
        file_path = f"data/grid{GRID_SIZE}/{assumed_behavior}/langrange_values/env_{k}.json"
        # with open(file_path, "r") as json_file:
        #     data = json.load(json_file)
        budget_buckets_realized = [[0,0] for _ in max_budgets]
        budget_buckets_wcd_change = [0] * len(max_budgets)
        budget_buckets_times = [0] * len(max_budgets)
        budget_best_lambdas = [0] * len(max_budgets)
        if  os.path.exists(file_path):
            with open(file_path, "r") as json_file:
                data = json.load(json_file)
            for lambda_pair in data["lambda_pairs"]:
                for i, budget in enumerate(max_budgets):

                    num_changes = lambda_pair["num_changes"]
                    wcd_change = lambda_pair["wcd_change"]
                    time_taken = lambda_pair["time_taken"]
                    lambdas = lambda_pair["lambdas"]

                    if not (lambdas[0] in lambda1_values and lambdas[1] in lambda2_values):
                        continue

                    if np.sum(num_changes) <= budget:
                        if wcd_change >=budget_buckets_wcd_change[i]:  # found a better value
                            budget_buckets_wcd_change[i] = wcd_change
                            budget_buckets_realized[i] = num_changes
                            budget_buckets_times[i] = time_taken
                            budget_best_lambdas[i]= lambdas
                            # print(budget,lambdas, wcd_change)
        else:
            print(f"env_{k}.json missing")
            # continue
            budget_buckets_times = [700] * len(max_budgets)

        # Append results for current k to the lists of all results
        all_wcd_changes.append(budget_buckets_wcd_change)
        all_budgets_realized.append(budget_buckets_realized)
        all_times.append(budget_buckets_times)
        all_best_lambdas.append(budget_best_lambdas)

        # print(budget_best_lambdas)

    # Save results to CSV files
    n_lambda = len(lambda_values)
    data_storage_path = f"{data_storage_path}/n_lambdas_{n_lambda}"
    create_folder(data_storage_path)
    create_or_update_list_file(f"{data_storage_path}/times_{GRID_SIZE}_{experiment_label}.csv", all_times)
    create_or_update_list_file(f"{data_storage_path}/wcd_change_{GRID_SIZE}_{experiment_label}.csv", all_wcd_changes)
    create_or_update_list_file(f"{data_storage_path}/budgets_{GRID_SIZE}_{experiment_label}.csv", all_budgets_realized)
    create_or_update_list_file(f"{data_storage_path}/max_budgets_{GRID_SIZE}_{experiment_label}.csv",[max_budgets])
    create_or_update_list_file(f"{data_storage_path}/best_lambdas_{GRID_SIZE}_{experiment_label}.csv",all_best_lambdas)
    # print(data_storage_path)
    return all_wcd_changes, all_budgets_realized, all_times

# Example usage
# k_values = range(0, 1, 14)
k_values = range(0, max_id, interval)
# k_values = [16080]
lambda_values = [0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2, 5, 7]
max_budgets = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23,25,27,29,31,33,35,37,39,41]

experiment_label =f"{assumed_behavior}_BOTH_UNIFORM_test"

all_wcd_changes, all_budgets_realized, all_times = process_experiment(k_values, lambda_values, experiment_label,GRID_SIZE=GRID_SIZE)

# %% [code] cell 9
temp_values = np.array(lambda_values)
wcds_by_n = []
ns = []
i = 0
while True:
    all_wcd_changes, all_budgets_realized, all_times = process_experiment(k_values, temp_values, experiment_label,GRID_SIZE=GRID_SIZE)
    wcds_by_n.append(np.mean(all_wcd_changes, axis=0))
    ns.append(temp_values.shape[0]**2)
    print(temp_values)
    temp_values *= 10
    temp_values = temp_values[temp_values <= 10]


    i += 1
    if i >= 5:
        break

# %% [code] cell 10
# Plotting the curves
for i, wcds in enumerate(wcds_by_n):
    print(i)
    plt.plot(max_budgets,wcds, label=f'n={ns[i]}')

# Add labels and legend
plt.xlabel('Budget')
plt.ylabel('WCD Change')
plt.title('Result from different number of langrange pairs')
plt.legend()

# Show plot
plt.show()

# %% [code] cell 11
assumed_behavior = "HUMAN"

# %% [code] cell 12
def combine_environments(k_values, lambda_values,experiment_label, GRID_SIZE=10):
    """Process the experiment with given parameters."""
    # Define lists to store results
    all_wcd_changes = []
    all_budgets_realized = []
    all_times = []
    data_storage_path = f"baselines/data/grid{GRID_SIZE}/timeout_600/{assumed_behavior}/{experiment_label}"

    blocking_rat = 1.05
    unblocking_rat = 1
    data_storage_path = f"baselines/data/grid{GRID_SIZE}/timeout_600/{assumed_behavior}/{experiment_label}"

    lambda1_values = lambda_values
    lambda2_values = lambda_values
    # Process data for each k
    for k in k_values:
        # Read JSON file for current k
        file_path = f"baselines/data/grid{GRID_SIZE}/timeout_600/{assumed_behavior}/{experiment_label}/individual_envs/env_{k}.json"
        with open(file_path, "r") as json_file:
            data = json.load(json_file)


        all_wcd_changes.append(data["wcd_changes"])
        all_budgets_realized.append(data["num_changes"])
        all_times.append(data["times"])
        # max_budgets = np.sum(data["max_budgets"], axis = 1)+1
        max_budgets = data["max_budgets"]

    # Save results to CSV files
    n_lambda = len(lambda_values)
    create_or_update_list_file(f"{data_storage_path}/times_{GRID_SIZE}_{experiment_label}.csv", all_times)
    create_or_update_list_file(f"{data_storage_path}/wcd_change_{GRID_SIZE}_{experiment_label}.csv", all_wcd_changes)
    create_or_update_list_file(f"{data_storage_path}/budgets_{GRID_SIZE}_{experiment_label}.csv", all_budgets_realized)
    create_or_update_list_file(f"{data_storage_path}/num_changes_{GRID_SIZE}_{experiment_label}.csv", all_budgets_realized)
    create_or_update_list_file(f"{data_storage_path}/max_budgets_{GRID_SIZE}_{experiment_label}.csv",[max_budgets])
    return all_wcd_changes, all_budgets_realized, all_times

# Example usage
k_values = range(0, max_id, interval)
max_budgets = []

experiment_label ="BOTH_UNIFORM_GREEDY_TRUE_WCD"

all_wcd_changes, all_budgets_realized, all_times = combine_environments(k_values, lambda_values, experiment_label,GRID_SIZE=GRID_SIZE)
