"""
Baseline Experiments Script for Overcooked-AI Goal Recognition Design

Author: Robert Kasumba (rkasumba@wustl.edu)

This script implements baseline methods for comparison with our optimization approach.
It includes greedy algorithms that use either predicted WCD (from CNN oracle) or true WCD
(computed via oracle simulation) to guide environment modifications.

WHY THIS IS NEEDED:
- Provides baseline comparisons to evaluate our optimization approach
- Implements greedy strategies that are commonly used in similar problems
- Tests both predicted and true WCD to assess CNN oracle quality
- Enables fair comparison of different optimization strategies

HOW IT WORKS:
1. Loads environment dataset and trained CNN oracle (if using predicted WCD)
2. For each environment, applies greedy modification strategies:
   - Greedy with predicted WCD: Uses CNN oracle for fast WCD estimation
   - Greedy with true WCD: Uses oracle simulation for exact WCD computation
3. Supports both constrained and unconstrained modification scenarios
4. Tracks optimization progress, WCD changes, and computation time
5. Saves results for comparison with our gradient-based approach

BASELINE METHODS:
- GREEDY_PRED_WCD: Greedy optimization using CNN-predicted WCD values
- GREEDY_TRUE_WCD: Greedy optimization using true WCD from oracle simulation
- Both methods support constrained (ratio-based) and unconstrained scenarios

USAGE:
    python run_baseline_experiments.py --cost 10 --max_grid_size 6 --experiment_label test --experiment_type GREEDY_TRUE_WCD --optimality OPTIMAL --start_index 0 --timeout_seconds 18000 --ratio 1_3

OUTPUT:
    Baseline results in ./src/overcooked_ai_py/baselines directory
"""

import sklearn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


import torch
import ast
import pickle
import utils
import os
import pdb
import json
import torch.nn as nn
import random
import argparse
import traceback
import time


plt.figure(dpi=150)
from utils import *
from wcd_simulation_utils import *
from func_timeout import func_timeout, FunctionTimedOut

from func_timeout import func_timeout, FunctionTimedOut
import time

def explore_multiple_changes_with_timeout(x, grid_size=6, init_wcd=0, pred_model=None, use_true_wcd=False, timeout=30, max_changes=[-1, -1]):
    """
    Greedy baseline optimization with timeout handling and constraint management.
    
    Args:
        x: Initial environment tensor
        grid_size: Size of the gridworld
        init_wcd: Initial WCD value
        pred_model: Trained CNN oracle for WCD prediction (if use_true_wcd=False)
        use_true_wcd: Whether to use true WCD (oracle simulation) or predicted WCD
        timeout: Maximum time allowed for optimization
        max_changes: Maximum allowed changes [OT_changes, SPD_changes]
    
    Returns:
        (best_x, best_wcd): Best environment and its WCD value
        elapsed_time: Time taken for optimization
        final_wcd: Final WCD value achieved
    """
    start_time = time.time()
    try:
        cur_x = x.clone()  # Current environment state
        prev_wcd = init_wcd  # Previous WCD value
        print("======= Initial MAX CHANGES=====", max_changes)
        
        # Greedy optimization loop
        while True:
            # Try to find the next best change within timeout
            new_x, new_wcd = func_timeout(timeout, explore_changes, 
                                        args=(cur_x, grid_size, prev_wcd, pred_model, use_true_wcd, 
                                             compute_next_change(max_changes), True))
            
            # Compute actual changes made
            changes = compute_changes(decode_env(cur_x.squeeze()), decode_env(new_x.squeeze()))
            changes = [changes["O+T"], changes["S+P+D"]]  # Parse into OT and SPD components
            print(" CHANGES : ", changes)
            
            # Check if changes exceed maximum allowed
            if (np.array(changes) > np.array(max_changes)).any(): 
                break
                
            # Update remaining budget
            new_max_changes = np.array(max_changes) - np.array(changes)
            print("Previous change was", changes, "MAX CHANGES is now at", new_max_changes)
            max_changes = np.clip(new_max_changes, a_min=0, a_max=np.max(max_changes))
            prev_wcd = new_wcd 
            
            # Check termination conditions
            if (cur_x == new_x).all() or np.sum(max_changes) == 0:
                cur_x = new_x
                break
            cur_x = new_x
            
        elapsed_time = time.time() - start_time
        return (cur_x, prev_wcd), elapsed_time, prev_wcd
        
    except FunctionTimedOut:
        # Handle timeout gracefully
        elapsed_time = time.time() - start_time
        print(f"Function timed out after {elapsed_time:.2f} seconds.")
        # Return the preliminary values on timeout
        return (x, init_wcd), elapsed_time, init_wcd
    

def explore_shared_budget_with_timeout(x, grid_size=6,init_wcd=0,  pred_model=None, use_true_wcd=False, timeout= 30,max_changes =0):
    start_time = time.time()
    try:
        cur_x = x.clone()
        prev_wcd = init_wcd
        print("======= Initial MAX CHANGES=====",max_changes)
        while True:
            new_x,new_wcd = func_timeout(timeout, explore_changes, args=(cur_x, grid_size,prev_wcd, pred_model, use_true_wcd,1,False))
            
            changes = compute_changes(decode_env(cur_x.squeeze()), decode_env(new_x.squeeze()))
            changes=[changes["O+T"],changes["S+P+D"]]
            print(" CHANGES : ", changes)
            
            if np.sum(changes)>max_changes: 
                break
             
            new_max_changes = max_changes - np.sum(changes)
            
            print("Previous change was",np.sum(changes),"MAX CHANGES is now at",new_max_changes)
            max_changes = np.clip(new_max_changes, a_min = 0,a_max =np.max([0,max_changes]))
            prev_wcd = new_wcd 
            
            if (cur_x == new_x).all() or np.sum(max_changes) ==0:
                cur_x = new_x
                break
            cur_x = new_x
            
        elapsed_time = time.time() - start_time
 
        return (cur_x,prev_wcd), elapsed_time, prev_wcd
                
    except FunctionTimedOut:
        elapsed_time = time.time() - start_time
        print(f"Function timed out after {elapsed_time:.2f} seconds.")
        # Return the preliminary values on timeout
        return (x, init_wcd), elapsed_time, init_wcd  # Return init_wcd again to signify that it's the last known value

    
def explore_changes_with_timeout(x, grid_size=6,init_wcd=0,  pred_model=None, use_true_wcd=False, timeout= 30,max_changes =[-1,-1]):
    start_time = time.time()
    try:
        result = func_timeout(timeout, explore_changes, args=(x, grid_size,init_wcd, pred_model, use_true_wcd,max_changes))
        elapsed_time = time.time() - start_time
        _, cur_wcd = result
        return result, elapsed_time, cur_wcd
    except FunctionTimedOut:
        elapsed_time = time.time() - start_time
        print(f"********* Function TIMEDOUT after {elapsed_time:.2f} seconds.*******")
        # Return the preliminary values on timeout
        return (x, init_wcd), elapsed_time, init_wcd  # Return init_wcd again to signify that it's the last known value

def compute_next_change(max_changes):
    change = []
    if max_changes[0] > 0 and max_changes[1] > 0:
        change = [1, 1]
    else:
        change = [1 if max_changes[0] > 0 else 0, 1 if max_changes[1] > 0 else 0]
    return change

def iterative_greedy_search_with_timeout(initial_x,  budget=5, grid_size=6, pred_model=None, use_true_wcd=False, timeout = 30, ratio = "1_2"):
    current_x = initial_x.clone()
    total_elapsed_time = 0
    
    budgets = []
    init_wcd = compute_true_wcd(initial_x, grid_size=grid_size)
    cur_wcd = init_wcd
    env_i_s =[initial_x]
    env_true_wcd = [init_wcd]
    
    budget_times= []
    budget_num_changes = []
    budget_wcd_change = []
    print("Init WCD = ",cur_wcd)
    
    previous_wcd = cur_wcd
    counter_no_change = 0
    bugdets = list(range(1,budget+1,2))
    max_no_change = 2
    for iteration in bugdets:
        print(f"Iteration {iteration}:") #TO DO remove the 4
        # Perform one iteration of greedy search with timeout
        try:
            b = iteration
            
            if ratio =="0_0":
          # Calculating number of items
                (new_x, cur_wcd), elapsed_time, last_valid_wcd = explore_shared_budget_with_timeout(initial_x.clone(), grid_size, init_wcd, 
                                                                                              pred_model, use_true_wcd, timeout/budget,b)
            else:
                ratio = ratio #ot to spd"
                ratio_ot, ratio_spd = map(int, ratio.split('_'))  # Extracting ratio values
                
                max_no_change = np.max([ratio_ot, ratio_spd]) # the maximum number of no changes before we terminate - for ratios, we try to capture as many budget

                num_ot, num_spd = round(b * ratio_ot / (ratio_ot + ratio_spd)), round(b * ratio_spd / (ratio_ot + ratio_spd))  # Calculating number of items
                (new_x, cur_wcd), elapsed_time, last_valid_wcd = explore_multiple_changes_with_timeout(initial_x.clone(), grid_size, init_wcd, 
                                                                                              pred_model, use_true_wcd, timeout/budget,[num_ot, num_spd])
            total_elapsed_time = elapsed_time
            
        except FunctionTimedOut:
            total_elapsed_time = timeout/budget
            print("Iteration timed out. Returning last valid WCD.")
            break

        
        budget_times.append(total_elapsed_time)
        changes = compute_changes(decode_env(initial_x.squeeze()), decode_env(new_x.squeeze()))
        budget_num_changes.append(changes)
        final_wcd = compute_true_wcd(new_x, grid_size=grid_size)
        budget_wcd_change.append(init_wcd - final_wcd)
        env_i_s.append(current_x.clone())
        env_true_wcd.append(final_wcd)
        
        print("Final WCD: ",final_wcd, "change is ",init_wcd - final_wcd)
        if previous_wcd == cur_wcd:
            counter_no_change +=1
        else:
            counter_no_change =0 # reset the counter
            
        previous_wcd = cur_wcd
        
        if elapsed_time>= (timeout/budget) or counter_no_change >=max_no_change:
            print("No further changes. Stopping.", "wcd",cur_wcd )
            break
        # Use the result of the current iteration as input to the next
        current_x = new_x.clone()
    
    
    if iteration<budget: # there was a timeout
        target_length = len(bugdets)
        # Filling up 'budget_times' and 'wcd_change' with their last value to match the length of 'budgets'
        budget_num_changes.extend([budget_num_changes[-1]] * (target_length - len(budget_num_changes)))
        budget_wcd_change.extend([budget_wcd_change[-1]] * (target_length - len(budget_wcd_change)))
        for _ in range((target_length - len(budget_times))):
            budget_times.extend([budget_times[-1]])
        

    return budget_num_changes, budget_wcd_change, budget_times, env_i_s,env_true_wcd




def get_wcd(x, true_wcd = True, model = None, grid_size = 6):
    
    if true_wcd:
        return compute_true_wcd(x, grid_size = grid_size)
    else:
        return model(x.cuda())


def compute_changes(env_init, env_final):
    """
    Compute the change in each object's count and Manhattan distance shift
    from the initial environment to the final environment.
    
    Args:
    - env_init (numpy.ndarray): Initial environment array
    - env_final (numpy.ndarray): Final environment array
    
    Returns:
    - changes (dict): Dictionary containing the change for each tracked object
    """
    # Count the number of 'X's in the initial and final environments
    num_X_init = np.sum(env_init == 'X')
    num_X_final = np.sum(env_final == 'X')

    # Compute the change in the number of 'X's
    delta_X = int(num_X_final - num_X_init)

    # Define the objects to track Manhattan distance shift
    objects = ['1', 'S', 'O', 'T', 'D', 'P']

    # Initialize a dictionary to store changes
    changes = {'X': delta_X}

    # Compute the change for each tracked object
    for obj in objects:
        # Find the coordinates of the object in the initial and final environments
        init_coords = np.argwhere(env_init == obj)
        final_coords = np.argwhere(env_final == obj)
        
        # Compute the Manhattan distance shift
        shift = np.sum(np.abs(init_coords - final_coords))
        changes[obj] = int(shift)
        
    changes["O+T"] = changes["O"]+changes["T"]
    changes["S+P+D"] = changes["S"]+changes["P"]+changes["D"]
                                          
    return changes

def explore_changes(x, grid_size=6, init_wcd=0, pred_model=None, use_true_wcd=False,max_changes =[-1,-1], individual_budget = True):
    rows, cols = grid_size, grid_size
    min_wcd = init_wcd  # Initialize with positive infinity
    x_i = x.clone()
    grid = extract_env_details(x, grid_size=grid_size)[0]
    env_init = grid.copy()
    best_x_i = x
    
    print("Max Change",max_changes, individual_budget)
    
    def explore_helper(i, j, new_i, new_j):
        nonlocal min_wcd, best_x_i, x_i
        
        # Swap positions
        grid[i][j], grid[new_i][new_j] = grid[new_i][new_j], grid[i][j]
        
        changes = compute_changes(env_init, grid)
        if individual_budget: 
            if (changes["O+T"]>max_changes[0]) or (changes["S+P+D"]>max_changes[1]) or ((changes["S+P+D"]+changes["O+T"])>1): # invali
                grid[i][j], grid[new_i][new_j] = grid[new_i][new_j], grid[i][j]
                #swap back if Invalid
                return None
        else: # total change should be 1
            assert np.sum(max_changes) <= 1, "Sum of max_changes exceeds 1"
                
            if (changes["O+T"]+changes["S+P+D"])>np.sum(max_changes): # invali
                grid[i][j], grid[new_i][new_j] = grid[new_i][new_j], grid[i][j]
                #swap back if Invalid
                return None
        
        if check_env_is_valid(grid):
            x_i[0, 0:8, :, :] = torch.tensor(encode_env(grid, grid_size=grid_size))
            wcd = get_wcd(x_i, true_wcd=use_true_wcd, model=pred_model, grid_size=grid_size)
            if wcd <= min_wcd:
                min_wcd = wcd
                best_x_i = x_i.clone()
        
        # Swap back the grid
        grid[i][j], grid[new_i][new_j] = grid[new_i][new_j], grid[i][j]
    
    
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] in ('S', 'T', 'P', 'O', 'D'): # one of the objects
                for new_i in range(rows):
                    for new_j in range(cols): # the new position is new_i, new_j
                        
                        # Swap the new position object with the object at i,j
                        if (i != new_i or j != new_j) and grid[new_i][new_j] != "1":
                            explore_helper(i, j, new_i, new_j)
            else:
                continue
                

    return best_x_i, min_wcd


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Simulate data")
    parser.add_argument(
        "--cost",
        type=float,  # Ensure that the input is expected to be a float
        default=0,  # Set the default value to 0
        help="Cost parameter for the simulation. Default is 0.",
    )
    
    parser.add_argument(
        "--max_grid_size",
        type=int,  # Ensure that the input is expected to be a int
        default=6,  # Set the default value to 1
        help="Maximum grid size.",
    )
    
    parser.add_argument(
    "--experiment_label",
    type=str,  # Ensure that the input is expected to be a string
    default="valonly",  # Set a default label for the experiment
    help="Label for the current experiment run. Default is 'default_experiment'.",
)
    
    parser.add_argument(
        "--experiment_type",
        type=str,  # Ensure that the input is expected to be a string
        default="GREEDY_TRUE_WCD_CONSTRAINED",  # Set a default label for the experiment
        choices =["GREEDY_TRUE_WCD","GREEDY_PRED_WCD","GREEDY_TRUE_WCD_CONSTRAINED","GREEDY_PRED_WCD_CONSTRAINED"],
        # choices =["ALL_MODS_EXHUASTIVE","ALL_MODS_GREEDY_TRUE_WCD","ALL_MODS_GREEDY_PRED_WCD"],
        help="Label for the current experiment run. Default is 'BLOCKING_ONLY_EXHUASTIVE'.",
    )
    
    parser.add_argument(
        "--optimality",
        type=str,  # Ensure that the input is expected to be a string
        default="OPTIMAL",  # Set a default label for the experiment
        choices =["OPTIMAL","SUBOPTIMAL"],
        # choices =["ALL_MODS_EXHUASTIVE","ALL_MODS_GREEDY_TRUE_WCD","ALL_MODS_GREEDY_PRED_WCD"],
        help="Behavior optimality'.",
    )
    
    parser.add_argument(
        "--start_index",
        type=int,  # Ensure that the input is expected to be a int
        default=0,  # Set the default value to 1
        help="Starting index for the environments",
    )
    
    parser.add_argument(
        "--timeout_seconds",
        type=int,  # Ensure that the input is expected to be a int
        default=18000,  # Set the default value to 1
        help="Timeout seconds",
    )
    
    parser.add_argument(
        "--ratio",
        type=str,  # Ensure that the input is expected to be a string
        default="1_3",  # Set a default label for the experiment
        choices =["1_2","1_3", "1_4","1_5","1_9","0_0","1_2","2_1"],
        # choices =["ALL_MODS_EXHUASTIVE","ALL_MODS_GREEDY_TRUE_WCD","ALL_MODS_GREEDY_PRED_WCD"],
        help="Ratio of ",
    )

    
    
    args = parser.parse_args()
    
    # Parse command line arguments
    cost = args.cost
    grid_size = args.max_grid_size
    base_data_path = f"./data/grid{grid_size}"
    experiment_label = args.experiment_label
    optimality = args.optimality
    
    # Set folder name based on agent optimality
    optim_folder = "optim_runs" if optimality == "OPTIMAL" else "suboptimal_runs"

    # Load environment dataset
    with open(f"{base_data_path}/model_training/dataset_{grid_size}_{args.experiment_label}.pkl", "rb") as f:
        loaded_dataset = pickle.load(f)
    
    # Load trained CNN oracle for predicted WCD experiments
    device = "cuda:0"
    model = torch.load(f"models/wcd_nn_oracle_random_{grid_size}_{args.experiment_label}.pt")
    model = model.to(device).eval()
    
    # Set up output directory for baseline results
    base_data_path = f"./baselines/data/grid{grid_size}"
    
    # Initialize tracking variables for results
    true_wcds_per_cost = []
    wcds_per_cost = []
    gammas = []
    interval = 2
    times_taken = []
    max_budgets = []
    wcd_changes = []
    budget_changes = []
    pred_wcds = []
    max_budget = 15
    gammas = []
    budgets = [i for i in range(1, max_budget + 1, 2)]  # Budget levels to test
    
    # Set timeout for optimization
    timeout = args.timeout_seconds
    
    # Determine whether to use true WCD or predicted WCD
    use_true_wcd = "TRUE_WCD" in args.experiment_type
    
    # Create output directories
    data_storage_path = f"{base_data_path}/{optim_folder}/timeout_{timeout}/{args.experiment_type}/ratio_{args.ratio}"
    create_folder(data_storage_path)
    create_folder(data_storage_path + f"/individual_envs")
    create_folder(data_storage_path + f"/individual_envs/final_envs")
    
    # Save budget configuration
    create_or_update_list_file(f"{data_storage_path}/max_budgets_{grid_size}_{args.experiment_type}.csv", [budgets])
    
    # Set up processing range
    max_index = len(loaded_dataset)
    interval = 20  # Process environments in batches of 20
    print("Interval ", interval)
    max_index = np.min([args.start_index + (interval * 500), len(loaded_dataset)])
    
    # Main baseline experiment loop
    for i in range(args.start_index, max_index, 20):
        try:
            print("Environment; ", i)
            x, y = loaded_dataset[i]  # Get a specific data sample
            
            # Set gamma value based on agent optimality
            if optimality == "OPTIMAL":
                gamma = 0.99999  # Near-optimal agent behavior
            else:
                gamma = randomly_choose_gamma(ranges=[(0.65, 1.0)], probabilities=[1.0])  # Suboptimal behavior
            
            # Set gamma value across all positions in the environment
            x[8, :, :] = torch.full((grid_size, grid_size), gamma).float()
            x = x.unsqueeze(0).float().cuda()  # Add batch dimension and move to GPU
            
            print(x.squeeze(0).shape)
            print("Original X:, WCD = ", model(x).item())
            
            # Initialize tracking variables for this environment
            best_wcd = []
            invalid_envs_collection = []
            true_wcds = []
            budget_time_taken = []
            budget = []
            wcd_change = []
            num_changes = []
            
            # Run greedy baseline optimization
            budget_num_changes, budget_wcd_change, budget_times, envs_is, env_true_wcd = iterative_greedy_search_with_timeout(
                x, grid_size=grid_size, use_true_wcd=use_true_wcd, pred_model=model, 
                timeout=timeout, budget=max_budget, ratio=args.ratio)
                
            # Save intermediate results
            update_or_create_dataset(f"{base_data_path}/{optim_folder}/simulated_valids_final{grid_size}.pkl", envs_is, env_true_wcd)

            # Store initial and final environments for analysis
            update_or_create_dataset(f"{data_storage_path}/initial_envs_{grid_size}_{args.experiment_type}.pkl", 
                                     [envs_is[0]], [env_true_wcd[0]])  # Initial environment
            update_or_create_dataset(f"{data_storage_path}/final_envs_{grid_size}_{args.experiment_type}.pkl", 
                                     [envs_is[-1]], [env_true_wcd[-1]])  # Final environment

            # Store results for this environment
            times_taken.append(budget_times)
            wcd_changes.append(budget_wcd_change)
            budget_changes.append(budget_num_changes)
            gammas.append(gamma)
            
            env_dict = {
                    "env_id":i,
                    "times":budget_times,
                    "wcd_changes":budget_wcd_change,
                    "budgets":budget_num_changes,
                    "gamma":gamma,
                    "max_budgets":budgets,
                    "num_changes":budget_num_changes,
                    "final_env_name":f"env_{i}.pt"
                }
            # print(env_dict)
            with open(f"{data_storage_path}/individual_envs/env_{i}.json", "w") as json_file:
                json.dump(env_dict, json_file, indent=4)
            
            torch.save(envs_is[-1],f"{data_storage_path}/individual_envs/final_envs/env_{i}.pt")
        
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()
