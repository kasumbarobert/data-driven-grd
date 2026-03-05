import sys
import os
from pathlib import Path

sys.path.insert(0, "../")

from utils import *
from baseline_utils import *
from collections import deque
import time
import seaborn as sns
import multiprocessing
import argparse
from func_timeout import func_timeout, FunctionTimedOut

COMPUTE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("COMPUTE_DEVICE", COMPUTE_DEVICE, torch.cuda.is_available())

import json

def save_data_to_json(env_json_file_name, env_id, times, wcd_change, num_changes, init_wcd, y_env_by_budget, given_budget, meta_data ={}):
    """
    Save the provided data to a JSON file.

    Args:
        env_json_file_name (str): The name of the JSON file to save the data.
        env_id (int): Environment ID.
        times (list): List of budget times.
        wcd_change (list): List of WCD changes.
        num_changes (list): List of number of changes.
        init_wcd (float): Initial WCD value.
        y_env_by_budget (list): List of y_env_by_budget values.
        given_budget (List) : List of budget value
        meta_data (dict): Any extra details

    Returns:
        None
    """
    data = {
        "env_id": env_id,
        "meta_data":meta_data,
        "given_budget":given_budget,
        "times": times,
        "wcd_change": wcd_change,
        "num_changes": num_changes,
        "init_wcd": init_wcd,
        "y_env_by_budget": y_env_by_budget
    }

    # print(data)

    try:
        with open(env_json_file_name, 'w') as file:
            json.dump(data, file, indent=4)
        print(f"Env ID= {env_id} has been saved successfully")
    except Exception as e:
        print(f"An error occurred while saving data to JSON: {e}")

def run_search_all_modifications(root_grd_model,max_budget, all_mods_uniform ):
    start_time = time.time()
    final_grd = breadth_first_search_all_actions(root_grd_model, max_budget, all_mods_uniform=all_mods_uniform)
    end_time = time.time()

    time_taken = end_time - start_time
    wcd_diff = root_grd_model.get_wcd() - int(final_grd.get_wcd()) if final_grd is not None else None

    return final_grd, time_taken, wcd_diff

def run_search_greedy(root_grd_model,max_budget, all_mods_uniform):
    start_time = time.time()
    final_grd = greedy_search_true_wcd_all_mods(root_grd_model,max_budget,all_mods_uniform=all_mods_uniform)
    end_time = time.time()

    time_taken = end_time - start_time
    wcd_diff = root_grd_model.get_wcd() - int(final_grd.get_wcd()) if final_grd is not None else None

    return final_grd, time_taken, wcd_diff

def run_search_greedy_pred_wcd_all_mods(root_grd_model,max_budget, prediction_model,all_mods_uniform=False):
    # If you have multiple GPUs, replace '0' with the appropriate GPU index
    device = 'cpu'

    start_time = time.time()

    # Make sure any data passed to the model is on the correct device
    final_grd = greedy_search_predicted_wcd_all_mods(root_grd_model, prediction_model,max_budget,all_mods_uniform=all_mods_uniform)

    end_time = time.time()

    time_taken = end_time - start_time
    wcd_diff = root_grd_model.get_wcd() - int(final_grd.get_wcd()) if final_grd is not None else None

    # Move the result to CPU before putting it in the output queue
    return final_grd if final_grd is not None else None, time_taken, wcd_diff


    
def run_exhaustive_search_blocking(root_grd_model, max_budget,use_exhaustive_search = True):
    start_time = time.time()
    final_grd = breadth_first_search_blocking(root_grd_model,max_budget,use_exhaustive_search)
    end_time = time.time()

    time_taken = end_time - start_time
    wcd_diff = root_grd_model.get_wcd() - int(final_grd.get_wcd()) if final_grd is not None else None

    return final_grd, time_taken, wcd_diff

def can_skip_file(env_env_json_file_name):
    """
    Checks if a JSON file exists and has the expected number of keys.

    Parameters:
    - env_env_json_file_name (str): Path to the JSON file.

    Returns:
    - bool: True if the file exists
    """
    # Check if file exists
    if not os.path.exists(env_env_json_file_name):
        return False

    else:
        return True

def run_search_greedy_true_wcd_blocking(root_grd_model,max_budget):
    # If you have multiple GPUs, replace '0' with the appropriate GPU index

    start_time = time.time()
    print("Reached run_search_greedy_true_wcd_blocking")
    # Make sure any data passed to the model is on the correct device
    final_grd = greedy_search_true_wcd_blocking_only(root_grd_model, max_budget)

    end_time = time.time()

    time_taken = end_time - start_time
    wcd_diff = root_grd_model.get_wcd() - int(final_grd.get_wcd()) if final_grd is not None else None
    
    # Move the result to CPU before putting it in the output queue
    return final_grd if final_grd is not None else None, time_taken, wcd_diff

def run_search_greedy_pred_wcd_blocking_only(root_grd_model,max_budget, prediction_model):
    # If you have multiple GPUs, replace '0' with the appropriate GPU index
    device = 'cpu'
    start_time = time.time()

    # Make sure any data passed to the model is on the correct device
    final_grd = greedy_search_pred_wcd_blocking_only(root_grd_model, max_budget,prediction_model)

    end_time = time.time()

    time_taken = end_time - start_time

    # Move the result to CPU before putting it in the output queue
    return final_grd if final_grd is not None else None, time_taken, None

def run_all_modifications_exhuastive(grid_size = 6, dataset = None, experiment_type = "debug", verbose = False,env_run_index_seq=(0,28,14),
                                     timeout_seconds = 20,budgets=[1,2], data_storage_path=None,
                                    all_mods_uniform=False):
    times = []
    wcd_change = []
    num_changes =[]
    for i in range(env_run_index_seq[0], env_run_index_seq[1],env_run_index_seq[2]):
        env_json_file_name = f"{data_storage_path}/env_modifications/envs_{i}.json"
        if can_skip_file(env_json_file_name):
            print(f"Env iD = {i} already exists")
            continue
            
        if i>=len(loaded_dataset):
            print(f"Skipping index {i} as it exceeds the dataset size")
        x, y = loaded_dataset[i]  # Get a specific data sample
        x = x.unsqueeze(0).float().to(COMPUTE_DEVICE)
        grid = decode_grid_design(x[0].cpu(), return_map=True)
        grid_size, goal_positions, blocked_positions, start_pos,unblocked_positions = decode_grid_design(x[0].cpu())

        is_valid, cost_to_goals = is_design_valid(grid_size, goal_positions, blocked_positions, start_pos)
        if not is_valid:
            print("INVALID Original evironment")

        wcd,paths,wcd_paths = compute_wcd_single_env(grid_size, goal_positions, blocked_positions, 
                                                     start_pos, vis_paths = False, return_paths = True)

        budget_times = []
        budget_wcd_change = []
        budget_num_changes =[]
        x_env_by_budget = [x.cpu()]
        y_env_by_budget = [y.cpu().item()]
        for max_budget in budgets:
            
            root_grd_model = GRDModel( grid_size = grid_size, start_pos = start_pos, goal_positions = goal_positions,
                                  blocked_positions = blocked_positions, unblocked_positions = unblocked_positions,
                                 init_goal_costs = cost_to_goals, n_max_changes =max_budget )
            try:
                # Run the function with a specified timeout
                final_grd, time_taken, wcd_diff = func_timeout(timeout_seconds, 
                                                               run_search_all_modifications, args=(root_grd_model,max_budget,all_mods_uniform))
            except FunctionTimedOut:
                # Handle the timeout case
                print(f"The function exceeded the {timeout_seconds} second timeout.")
                final_grd, time_taken, wcd_diff =  None, timeout_seconds, None
            
                # print(int(final_grd.get_wcd()))
            budget_times.append(time_taken)

            init_grid = grid.copy()

            if not final_grd is None:
                final_grid = init_grid.copy()
                for b in final_grd.blocked_positions:
                    final_grid[b[0],b[1]] ="X"
                for b in final_grd.unblocked_positions:
                    final_grid[b[0],b[1]] =" "
            else:
                final_grid = init_grid
                final_grd = root_grd_model
                
            x_final = encode_from_grid_to_x(final_grid)
            
            budget_wcd_change.append(y.item()-int(final_grd.get_wcd()))
            
            x_changes = x_final.cpu()[:, 1, :, :]-x.cpu()[:, 1, :, :]
            
            blockings = (x_changes==1).sum(axis=(1, 2)).item()
            removals = (x_changes==-1).sum(axis=(1, 2)).item()
            
            budget_num_changes.append([blockings,removals])
            x_env_by_budget.append(x_final.cpu())
            y_env_by_budget.append(int(final_grd.get_wcd()))
            
            print("Budget",max_budget, "changes",[blockings,removals], "WCD", int(final_grd.get_wcd()),"Time:",time_taken)
            
            if time_taken == timeout_seconds: # no need to run higher budgets if smaller ones timeout
                target_length = len(budgets)
                # Filling up 'budget_times' and 'wcd_change' with their last value to match the length of 'budgets'
                budget_times.extend([budget_times[-1]] * (target_length - len(budget_times)))
                budget_wcd_change.extend([budget_wcd_change[-1]] * (target_length - len(budget_wcd_change)))
                budget_num_changes.extend([budget_num_changes[-1]] * (target_length - len(budget_num_changes)))
                x_env_by_budget.extend([x_env_by_budget[-1]] * (target_length - len(x_env_by_budget)))
                y_env_by_budget.extend([y_env_by_budget[-1]] * (target_length - len(y_env_by_budget)))
                break

        
        meta_data = {"grid_size":grid_size,"modification_method":experiment_type}
        save_data_to_json(env_json_file_name=env_json_file_name, env_id=i, times=budget_times, wcd_change=budget_wcd_change, 
                            num_changes=budget_num_changes, init_wcd=y_env_by_budget[0],
                            y_env_by_budget=y_env_by_budget, given_budget = budgets, meta_data = meta_data)
        
        update_or_create_dataset(f"{data_storage_path}/envs/envs_{i}_{grid_size}_{experiment_type}.pkl", x_env_by_budget, y_env_by_budget) # store the initial environments

def run_all_modifications_greedy_baseline_true_wcd(grid_size = 6, dataset = None, experiment_type = "debug", verbose = False,
                                                   env_run_index_seq=(0,28,14),timeout_seconds = 20, budgets =[1,10],data_storage_path=None,
                                                  all_mods_uniform=False):
    times = []
    wcd_change = []
    num_changes = []
    num_changes =[]
    
    for i in range(env_run_index_seq[0], env_run_index_seq[1],env_run_index_seq[2]):
        env_json_file_name = f"{data_storage_path}/env_modifications/envs_{i}.json"
        if can_skip_file(env_json_file_name):
            print(f"Env iD = {i} already exists")
            continue
        if i>=len(loaded_dataset):
            print(f"Skipping index {i} as it exceeds the dataset size")
        x, y = loaded_dataset[i]  # Get a specific data sample
        x = x.unsqueeze(0).float().to(COMPUTE_DEVICE)
        grid = decode_grid_design(x[0].cpu(), return_map=True)
        grid_size, goal_positions, blocked_positions, start_pos,unblocked_positions = decode_grid_design(x[0].cpu())

        is_valid, cost_to_goals = is_design_valid(grid_size, goal_positions, blocked_positions, start_pos)
        
        if not is_valid:
            print("INVALID Original evironment")
        wcd= compute_wcd_single_env_no_paths(grid_size, goal_positions, blocked_positions, 
                                                     start_pos, vis_paths = False, return_paths = True)

        budget_times = []
        budget_wcd_change = []
        budget_num_changes =[]
        x_env_by_budget = [x.cpu()]
        y_env_by_budget = [y.cpu().item()]
        
        for max_budget in budgets:
            
            root_grd_model = GRDModel( grid_size = grid_size, start_pos = start_pos, goal_positions = goal_positions,
                                  blocked_positions = blocked_positions, unblocked_positions = unblocked_positions,
                                 init_goal_costs = cost_to_goals,n_max_changes =max_budget ) 

            try:
                # Run the function with a specified timeout
                final_grd, time_taken, wcd_diff = func_timeout(timeout_seconds, 
                                                               run_search_greedy, args=(root_grd_model,max_budget,all_mods_uniform))
            except FunctionTimedOut:
                # Handle the timeout case
                print(f"The function exceeded the {timeout_seconds} second timeout.")
                final_grd, time_taken, wcd_diff =  None, timeout_seconds, None
            
            budget_times.append(time_taken)

            init_grid = grid.copy()

            if not final_grd is None:
                final_grid = init_grid.copy()
                for b in final_grd.blocked_positions:
                    if final_grid[b[0],b[1]]=="S":
                        print("WE HAVE A PROBLEM in blocked positions")
                        print("FINAL",final_grd.blocked_positions)
                        print("Root",root_grd_model.blocked_positions)
                        
                    final_grid[b[0],b[1]] ="X"
                for b in final_grd.unblocked_positions:
                    if final_grid[b[0],b[1]]=="S":
                        print("WE HAVE A PROBLEM in unblocking")
                        
                    final_grid[b[0],b[1]] =" "
            else:
                final_grid = init_grid
                final_grd = root_grd_model
            
            x_final = encode_from_grid_to_x(final_grid)
        
            budget_wcd_change.append(y.item()-int(final_grd.get_wcd()))
            
            x_changes = x_final.cpu()[:, 1, :, :]-x.cpu()[:, 1, :, :]
            blockings = (x_changes==1).sum(axis=(1, 2)).item()
            removals = (x_changes==-1).sum(axis=(1, 2)).item()
            
            budget_num_changes.append([blockings,removals])
            
            x_env_by_budget.append(x_final.cpu())
            y_env_by_budget.append(int(final_grd.get_wcd()))
            
            print("Budget", max_budget, " Used", budget_num_changes[-1], "WCD",y_env_by_budget[-1])
            
            if time_taken == timeout_seconds: # no need to run higher budgets if smaller ones timeout
                target_length = len(budgets)
                # Filling up 'budget_times' and 'wcd_change' with their last value to match the length of 'budgets'
                budget_times.extend([budget_times[-1]] * (target_length - len(budget_times)))
                budget_wcd_change.extend([budget_wcd_change[-1]] * (target_length - len(budget_wcd_change)))
                budget_num_changes.extend([budget_num_changes[-1]] * (target_length - len(budget_num_changes)))
                x_env_by_budget.extend([x_env_by_budget[-1]] * (target_length - len(x_env_by_budget)))
                y_env_by_budget.extend([y_env_by_budget[-1]] * (target_length - len(y_env_by_budget)))
                break

        meta_data = {"grid_size":grid_size,"modification_method":experiment_type}
        save_data_to_json(env_json_file_name=env_json_file_name, env_id=i, times=budget_times, wcd_change=budget_wcd_change, 
                            num_changes=budget_num_changes, init_wcd=y_env_by_budget[0],
                            y_env_by_budget=y_env_by_budget, given_budget = budgets, meta_data = meta_data)
        # (, , , , , , , )
        
        update_or_create_dataset(f"{data_storage_path}/envs/envs_{i}_{grid_size}_{experiment_type}.pkl", x_env_by_budget, y_env_by_budget) # store the initial environments

def run_all_modifications_greedy_baseline_pred_wcd(grid_size = 6, dataset = None, experiment_type = "debug", 
                                                   verbose = False, prediction_model = None, env_run_index_seq=(0,28,14),timeout_seconds = 20, 
                                                   budgets = [1,2], data_storage_path=None, all_mods_uniform=False):
    times = []
    wcd_change = []
    num_changes =[]
    for i in range(env_run_index_seq[0], env_run_index_seq[1],env_run_index_seq[2]):
        env_json_file_name = f"{data_storage_path}/env_modifications/envs_{i}.json"
        if can_skip_file(env_json_file_name):
            print(f"Env iD = {i} already exists")
            continue

        if i>=len(loaded_dataset):
            print(f"Skipping index {i} as it exceeds the dataset size")
        x, y = loaded_dataset[i]  # Get a specific data sample
        x = x.unsqueeze(0).float().to(COMPUTE_DEVICE)
        grid = decode_grid_design(x[0].cpu(), return_map=True)
        grid_size, goal_positions, blocked_positions, start_pos,unblocked_positions = decode_grid_design(x[0].cpu())

        is_valid, cost_to_goals = is_design_valid(grid_size, goal_positions, blocked_positions, start_pos)
        if not is_valid:
            print("INVALID Original evironment")

        wcd = compute_wcd_single_env_no_paths(grid_size, goal_positions, blocked_positions, start_pos, vis_paths = False, return_paths = True)

        budget_times = []
        budget_wcd_change = []
        budget_num_changes =[]
        x_env_by_budget = [x.cpu()]
        y_env_by_budget = [y.cpu().item()]
        
        for max_budget in budgets:
            root_grd_model = GRDModel( grid_size = grid_size, start_pos = start_pos, goal_positions = goal_positions,
                          blocked_positions = blocked_positions, unblocked_positions = unblocked_positions,
                         init_goal_costs = cost_to_goals, compute_wcd = False, n_max_changes =max_budget )
            try:
        # Run the function with a specified timeout
                final_grd, time_taken, wcd_diff = func_timeout(timeout_seconds, run_search_greedy_pred_wcd_all_mods, 
                                      args=(root_grd_model,max_budget,prediction_model,all_mods_uniform))
            except FunctionTimedOut:
                # Handle the timeout case
                print(f"The function exceeded the {timeout_seconds} second timeout.")
                final_grd, time_taken, wcd_diff=  None, timeout_seconds, None
            
            init_grid = grid.copy()

            if not (final_grd is None):
                final_grid = init_grid.copy()
                for b in final_grd.blocked_positions:
                    final_grid[b[0],b[1]] ="X"
                for b in final_grd.unblocked_positions:
                    final_grid[b[0],b[1]] =" "
            else:
                final_grid = init_grid
                final_grd = root_grd_model
                
            x_final = encode_from_grid_to_x(final_grid)
            
            budget_times.append(time_taken)
            budget_wcd_change.append(y.item()-int(final_grd.get_wcd()))
            x_changes = x_final.cpu()[:, 1, :, :]-x.cpu()[:, 1, :, :]
            blockings = (x_changes==1).sum(axis=(1, 2)).item()
            removals = (x_changes==-1).sum(axis=(1, 2)).item()
            
            budget_num_changes.append([blockings,removals])
            
            x_env_by_budget.append(x_final.cpu())
            y_env_by_budget.append(int(final_grd.get_wcd()))
            
            print("Budget", max_budget, " Used", budget_num_changes[-1], "WCD",y_env_by_budget[-1])
            if time_taken == timeout_seconds: # no need to run higher budgets if smaller ones timeout
                target_length = len(budgets)
                # Filling up 'budget_times' and 'wcd_change' with their last value to match the length of 'budgets'
                budget_times.extend([budget_times[-1]] * (target_length - len(budget_times)))
                budget_wcd_change.extend([budget_wcd_change[-1]] * (target_length - len(budget_wcd_change)))
                budget_num_changes.extend([budget_num_changes[-1]] * (target_length - len(budget_num_changes)))
                x_env_by_budget.extend([x_env_by_budget[-1]] * (target_length - len(x_env_by_budget)))
                y_env_by_budget.extend([y_env_by_budget[-1]] * (target_length - len(y_env_by_budget)))
                break
                
        meta_data = {"grid_size":grid_size,"modification_method":experiment_type}
        save_data_to_json(env_json_file_name=env_json_file_name, env_id=i, times=budget_times, wcd_change=budget_wcd_change, 
                            num_changes=budget_num_changes, init_wcd=y_env_by_budget[0],
                            y_env_by_budget=y_env_by_budget, given_budget = budgets, meta_data = meta_data)
        # (, , , , , , , )
        
        update_or_create_dataset(f"{data_storage_path}/envs/envs_{i}_{grid_size}_{experiment_type}.pkl", x_env_by_budget, y_env_by_budget) # store the initial environments
        
def run_blocking_only_baseline(grid_size = 6, dataset = None, experiment_type = "debug", use_exhaustive_search = False, verbose = False,
                               env_run_index_seq=(0,28,14),timeout_seconds = 20,budgets=[1,2],data_storage_path=None):
    times = []
    wcd_change = []
    num_changes =[]
    for i in range(env_run_index_seq[0], env_run_index_seq[1],env_run_index_seq[2]):
        env_json_file_name = f"{data_storage_path}/env_modifications/envs_{i}.json"
        if can_skip_file(env_json_file_name):
            print(f"Env iD = {i} already exists")
            continue
        if i>=len(loaded_dataset):
            print(f"Skipping index {i} as it exceeds the dataset size")
        x, y = loaded_dataset[i]  # Get a specific data sample
        x = x.unsqueeze(0).float().to(COMPUTE_DEVICE)
        grid = decode_grid_design(x[0].cpu(), return_map=True)
        grid_size, goal_positions, blocked_positions, start_pos,space_pos = decode_grid_design(x[0].cpu())

        try:
            wcd,paths,wcd_paths = func_timeout(timeout_seconds,compute_wcd_single_env, args=(grid_size, 
                                                                                             goal_positions, blocked_positions, 
                                                                                             start_pos, False, True))
            timed_out = False
        except FunctionTimedOut:
                # Handle the timeout case
                print(f"The function exceeded the {timeout_seconds} second timeout.")
                timed_out = True
        
        if not timed_out:
            root_grd_model = GRDModelBlockingOnly( grid, paths,blocked = [],fixed_positions = (grid_size, goal_positions, start_pos))
            root_grd_model.init_goal_costs = root_grd_model.get_shortest_paths()
            
            budget_times = []
            budget_wcd_change = []
            budget_num_changes =[]
            x_env_by_budget = [x.cpu()]
            y_env_by_budget = [y.cpu().item()]

            for max_budget in budgets:
                try:
                    # Run the function with a specified timeout
                    final_grd, time_taken, wcd_diff = func_timeout(timeout_seconds, 
                                                                   run_exhaustive_search_blocking, 
                                                                   args=(root_grd_model,max_budget,use_exhaustive_search))
                except FunctionTimedOut:
                    # Handle the timeout case
                    print(f"The function exceeded the {timeout_seconds} second timeout.")
                    final_grd, time_taken, wcd_diff =  None, timeout_seconds, None

                if not final_grd is None:
                    grid = final_grd.get_grid()
                    for b in final_grd.blocked:
                        grid[b[0],b[1]] ="X"
                else:
                    grid = root_grd_model.grid
                    final_grd = root_grd_model

                x_final = encode_from_grid_to_x(grid)


                budget_times.append(time_taken)
                budget_wcd_change.append(y.item()-int(final_grd.get_wcd()))

                x_changes = x_final.cpu()[:, 1, :, :]-x.cpu()[:, 1, :, :]
                blockings = (x_changes==1).sum(axis=(1, 2)).item()
                removals = (x_changes==-1).sum(axis=(1, 2)).item()

                budget_num_changes.append([blockings,removals])

                x_env_by_budget.append(x_final.cpu())
                y_env_by_budget.append(int(final_grd.get_wcd()))

                print("Budget", max_budget, " Used", budget_num_changes[-1], "WCD",y_env_by_budget[-1])

                if time_taken == timeout_seconds: # no need to run higher budgets if smaller ones timeout
                    target_length = len(budgets)
                    # Filling up 'budget_times' and 'wcd_change' with their last value to match the length of 'budgets'
                    budget_times.extend([budget_times[-1]] * (target_length - len(budget_times)))
                    budget_wcd_change.extend([budget_wcd_change[-1]] * (target_length - len(budget_wcd_change)))
                    budget_num_changes.extend([budget_num_changes[-1]] * (target_length - len(budget_num_changes)))
                    x_env_by_budget.extend([x_env_by_budget[-1]] * (target_length - len(x_env_by_budget)))
                    y_env_by_budget.extend([y_env_by_budget[-1]] * (target_length - len(y_env_by_budget)))
                    break
            # pdb.set_trace()

            times.append(budget_times)
            wcd_change.append(budget_wcd_change)
            num_changes.append(budget_num_changes)

            update_or_create_dataset(f"{data_storage_path}/envs/envs_{i}_{grid_size}_{experiment_type}.pkl", 
                                     x_env_by_budget, y_env_by_budget) # store the environments
        else:
            budget_times = [timeout_seconds]*len(budgets)
            budget_wcd_change = [0]*len(budgets)
            budget_num_changes =[[0,0]]*len(budgets)
            
        meta_data = {"grid_size":grid_size,"modification_method":experiment_type}
        save_data_to_json(env_json_file_name=env_json_file_name, env_id=i, times=budget_times, wcd_change=budget_wcd_change, 
                            num_changes=budget_num_changes, init_wcd=y_env_by_budget[0],
                            y_env_by_budget=y_env_by_budget, given_budget = budgets, meta_data = meta_data)

def run_blocking_only_greedy_true_wcd(grid_size = 6, dataset = None, experiment_type = "debug", verbose = False,
                                      env_run_index_seq=(0,28,14),timeout_seconds = 20, budgets =[1,2],data_storage_path=None):
    times = []
    wcd_change = []
    num_changes =[]

    for i in range(env_run_index_seq[0], env_run_index_seq[1],env_run_index_seq[2]):
        env_json_file_name = f"{data_storage_path}/env_modifications/envs_{i}.json"
        if can_skip_file(env_json_file_name):
            print(f"Env iD = {i} already exists")
            continue
        if i>=len(loaded_dataset):
            print(f"Skipping index {i} as it exceeds the dataset size")
        x, y = loaded_dataset[i]  # Get a specific data sample
        x = x.unsqueeze(0).float().to(COMPUTE_DEVICE)
        grid = decode_grid_design(x[0].cpu(), return_map=True)
        grid_size, goal_positions, blocked_positions, start_pos,space_pos = decode_grid_design(x[0].cpu())
        try:
            wcd,paths,wcd_paths = func_timeout(timeout_seconds,compute_wcd_single_env, args=(grid_size, 
                                                                                             goal_positions, blocked_positions, 
                                                                                             start_pos, False, True))
            timed_out = False
        except FunctionTimedOut:
                # Handle the timeout case
                print(f"The function exceeded the {timeout_seconds} second timeout.")
                timed_out = True
        
        if not timed_out:
            root_grd_model = GRDModelBlockingOnly( grid, paths,blocked = [], fixed_positions = (grid_size, goal_positions, start_pos))
            root_grd_model.init_goal_costs = root_grd_model.get_shortest_paths()

            budget_times = []
            budget_wcd_change = []
            budget_num_changes =[]
            x_env_by_budget = [x.cpu()]
            y_env_by_budget = [y.cpu().item()]

            for max_budget in budgets:

                try:
                    # Run the function with a specified timeout
                    final_grd, time_taken, wcd_diff = func_timeout(timeout_seconds, run_search_greedy_true_wcd_blocking, 
                                                                   args=(root_grd_model,max_budget))
                except FunctionTimedOut:
                    # Handle the timeout case
                    print(f"The function exceeded the {timeout_seconds} second timeout.")
                    final_grd, time_taken, wcd_diff =  None, timeout_seconds, None

                if not final_grd is None:
                    grid = final_grd.get_grid()
                    for b in final_grd.blocked:
                        grid[b[0],b[1]] ="X"
                else:
                    grid = root_grd_model.grid
                    final_grd = root_grd_model
                x_final = encode_from_grid_to_x(grid)

                budget_times.append(time_taken)
                budget_wcd_change.append(y.item()-int(final_grd.get_wcd()))

                x_changes = x_final.cpu()[:, 1, :, :]-x.cpu()[:, 1, :, :]
                blockings = (x_changes==1).sum(axis=(1, 2)).item()
                removals = (x_changes==-1).sum(axis=(1, 2)).item()

                budget_num_changes.append([blockings,removals])

                x_env_by_budget.append(x_final.cpu())
                y_env_by_budget.append(int(final_grd.get_wcd()))

                if time_taken == timeout_seconds: # no need to run higher budgets if smaller ones timeout
                    target_length = len(budgets)
                    # Filling up 'budget_times' and 'wcd_change' with their last value to match the length of 'budgets'
                    budget_times.extend([budget_times[-1]] * (target_length - len(budget_times)))
                    budget_wcd_change.extend([budget_wcd_change[-1]] * (target_length - len(budget_wcd_change)))
                    budget_num_changes.extend([budget_num_changes[-1]] * (target_length - len(budget_num_changes)))

                    x_env_by_budget.extend([x_env_by_budget[-1]] * (target_length - len(x_env_by_budget)))
                    y_env_by_budget.extend([y_env_by_budget[-1]] * (target_length - len(y_env_by_budget)))
                    break

            times.append(budget_times)
            wcd_change.append(budget_wcd_change)
            num_changes.append(budget_num_changes)
        
            update_or_create_dataset(f"{data_storage_path}/envs/envs_{i}_{grid_size}_{experiment_type}.pkl", 
                                     x_env_by_budget, y_env_by_budget) # store the environments
        else:
            budget_times = [timeout_seconds]*len(budgets)
            budget_wcd_change = [0]*len(budgets)
            budget_num_changes =[[0,0]]*len(budgets)
            
        env_json_file_name = f"{data_storage_path}/env_modifications/envs_{i}.json"
        meta_data = {"grid_size":grid_size,"modification_method":experiment_type}
        save_data_to_json(env_json_file_name=env_json_file_name, env_id=i, times=budget_times, wcd_change=budget_wcd_change, 
                            num_changes=budget_num_changes, init_wcd=y_env_by_budget[0],
                            y_env_by_budget=y_env_by_budget, given_budget = budgets, meta_data = meta_data)

def run_blocking_only_greedy_pred_wcd(grid_size = 6, dataset = None, experiment_type = "debug", verbose = False, 
                                      prediction_model = None, env_run_index_seq=(0,28,14),timeout_seconds = 20, budgets =[1,2],data_storage_path=None):
    times = []
    wcd_change = []
    num_changes =[]

    for i in range(env_run_index_seq[0], env_run_index_seq[1],env_run_index_seq[2]):
        env_json_file_name = f"{data_storage_path}/env_modifications/envs_{i}.json"
        if can_skip_file(env_json_file_name):
            print(f"Env iD = {i} already exists")
            continue
        if i>=len(loaded_dataset):
            print(f"Skipping index {i} as it exceeds the dataset size")
        x, y = loaded_dataset[i]  # Get a specific data sample
        x = x.unsqueeze(0).float().to(COMPUTE_DEVICE)
        grid = decode_grid_design(x[0].cpu(), return_map=True)
        grid_size, goal_positions, blocked_positions, start_pos,space_pos = decode_grid_design(x[0].cpu())

        try:
            wcd,paths,wcd_paths = func_timeout(timeout_seconds,compute_wcd_single_env, args=(grid_size, 
                                                                                             goal_positions, blocked_positions, 
                                                                                             start_pos, False, True))
            timed_out = False
        except FunctionTimedOut:
                # Handle the timeout case
                print(f"The function exceeded the {timeout_seconds} second timeout.")
                timed_out = True
        
        if not timed_out:

            root_grd_model = GRDModelBlockingOnly( grid, paths,blocked = [], compute_wcd = False,
                                                  fixed_positions = (grid_size, goal_positions, start_pos))
            root_grd_model.init_goal_costs = root_grd_model.get_shortest_paths()

            budget_times = []
            budget_wcd_change = []
            budget_num_changes =[]
            x_env_by_budget = [x.cpu()]
            y_env_by_budget = [y.cpu().item()]

            for max_budget in budgets:
                try:
                    # Run the function with a specified timeout
                    final_grd, time_taken, wcd_diff = func_timeout(timeout_seconds, run_search_greedy_pred_wcd_blocking_only, 
                                                                   args=(root_grd_model,max_budget,prediction_model))
                except FunctionTimedOut:
                    # Handle the timeout case
                    print(f"The function exceeded the {timeout_seconds} second timeout.")
                    final_grd, time_taken, wcd_diff = None, timeout_seconds, None

                init_grid = grid.copy()

                if not final_grd is None:
                    final_grid = init_grid.copy()
                    for b in final_grd.blocked:
                        grid[b[0],b[1]] ="X"
                else:
                    final_grid = init_grid
                    final_grd = root_grd_model
                x_final = encode_from_grid_to_x(grid)

                budget_times.append(time_taken)
                final_wcd = final_grd.compute_wcd(final_grd.paths)[0]
                budget_wcd_change.append(y.item()-final_wcd)

                x_changes = x_final.cpu()[:, 1, :, :]-x.cpu()[:, 1, :, :]
                blockings = (x_changes==1).sum(axis=(1, 2)).item()
                removals = (x_changes==-1).sum(axis=(1, 2)).item()

                budget_num_changes.append([blockings,removals])

                x_env_by_budget.append(x_final.cpu())
                y_env_by_budget.append(final_wcd)

                if time_taken == timeout_seconds: # no need to run higher budgets if smaller ones timeout
                    target_length = len(budgets)
                    # Filling up 'budget_times' and 'wcd_change' with their last value to match the length of 'budgets'
                    budget_times.extend([budget_times[-1]] * (target_length - len(budget_times)))
                    budget_wcd_change.extend([budget_wcd_change[-1]] * (target_length - len(budget_wcd_change)))
                    budget_num_changes.extend([budget_num_changes[-1]] * (target_length - len(budget_num_changes)))

                    x_env_by_budget.extend([x_env_by_budget[-1]] * (target_length - len(x_env_by_budget)))
                    y_env_by_budget.extend([y_env_by_budget[-1]] * (target_length - len(y_env_by_budget)))
                    break

            times.append(budget_times)
            wcd_change.append(budget_wcd_change)
            num_changes.append(budget_num_changes)

            update_or_create_dataset(f"{data_storage_path}/envs/envs_{i}_{grid_size}_{experiment_type}.pkl", 
                                     x_env_by_budget, y_env_by_budget) # store the environments
        else:
            budget_times = [timeout_seconds]*len(budgets)
            budget_wcd_change = [0]*len(budgets)
            budget_num_changes =[[0,0]]*len(budgets)
        
        meta_data = {"grid_size":grid_size,"modification_method":experiment_type}
        save_data_to_json(env_json_file_name=env_json_file_name, env_id=i, times=budget_times, wcd_change=budget_wcd_change, 
                            num_changes=budget_num_changes, init_wcd=y_env_by_budget[0],
                            y_env_by_budget=y_env_by_budget, given_budget = budgets, meta_data = meta_data)

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def parse_env_run_index_seq(env_run_index_seq):
    import re
    """
    Parses the env_run_index_seq string and validates its pattern.

    Args:
        env_run_index_seq (str): String in the format "startindex_endindex_interval".

    Returns:
        tuple: (startindex, endindex, interval) as integers.

    Raises:
        ValueError: If the input does not match the required pattern.
    """
    # Define the regex pattern for "startindex_endindex_interval"
    pattern = r"^(\d+)_(\d+)_(\d+)$"
    match = re.match(pattern, env_run_index_seq)

    if not match:
        raise ValueError(
            f"Invalid format for env_run_index_seq: '{env_run_index_seq}'. "
            "Expected format: 'startindex_endindex_interval', e.g., '0_10_2'."
        )

    # Extract components and convert them to integers
    startindex, endindex, interval = map(int, match.groups())
    return (startindex, endindex, interval)
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Simulate data")
    
    parser.add_argument(
        "--experiment_type",
        type=str,  # Ensure that the input is expected to be a string
        default="ALL_MODS_GREEDY_TRUE_WCD",  # Set a default label for the experiment
        choices =["BLOCKING_ONLY_EXHAUSTIVE", "BLOCKING_ONLY_PRUNE_REDUCE","BLOCKING_ONLY_GREEDY_TRUE_WCD","BLOCKING_ONLY_GREEDY_PRED_WCD","ALL_MODS_EXHAUSTIVE",
                  "ALL_MODS_GREEDY_TRUE_WCD","ALL_MODS_GREEDY_PRED_WCD","BOTH_UNIFORM_EXHAUSTIVE",
                  "BOTH_UNIFORM_GREEDY_PRED_WCD","BOTH_UNIFORM_GREEDY_TRUE_WCD"],
        # choices =["ALL_MODS_EXHUASTIVE","ALL_MODS_GREEDY_TRUE_WCD","ALL_MODS_GREEDY_PRED_WCD"],
        help="Label for the current experiment run. Default is 'BLOCKING_ONLY_EXHUASTIVE'.",
    )
    parser.add_argument(
        "--grid_size",
        type=int,  # Ensure that the input is expected to be a int
        default=13,  # Set the default value to 1
        help="Maximum grid size.",
    )
    
    parser.add_argument(
        "--env_run_index_seq",
        type=str,  # Ensure that the input is expected to be a str
        default="0_56_14",  # Set the default value to 1:1
        help="spacing in the test dataset to use for the experiment",
    )
    
    parser.add_argument(
        "--timeout_seconds",
        type=int,  # Ensure that the input is expected to be a int
        default=600,  # Set the default value to 1
        help="Timeout seconds",
    )
    
    parser.add_argument(
        "--ratio",
        type=str,  # Ensure that the input is expected to be a str
        default="1_3",  # Set the default value to 1:1
        # choices = ["ALL_MODS"],
        choices = ["1_1","1_2","1_3","3_1","1_5","5_1","2_1","3_2","7_1","9_1","1_7"], #1_3 for 6 and 1_5 for 13
        help="Either BLOCKING_ONLY or ALL_MODS for all modifications.",
    )
    parser.add_argument(
        "--wcd_pred_model_id",
        type=str,  # Ensure that the input is expected to be a str
        required = True,
        help="Trained Model ID",
    )

    args = parser.parse_args()
    experiment_type = args.experiment_type
    grid_size = args.grid_size
    env_run_index_seq = parse_env_run_index_seq(args.env_run_index_seq)
    timeout_seconds = args.timeout_seconds
    _label = "_best"
    total_budgets = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41]

    print(f"Running with the following configurations {args}")
    
    if "PRED_WCD" in args.experiment_type:
        # Ensure that `wcd_pred_model_id` is provided
        if not hasattr(args, 'wcd_pred_model_id') or not args.wcd_pred_model_id:
            raise ValueError("The predictive mode ID (wcd_pred_model_id) must be provided.")
        
        # Set the trained model directory and construct the model file path
        args.trained_model_dir = f"../models/wcd_prediction/grid{grid_size}/"
        model_label = f"{args.trained_model_dir}/training_logs/{args.wcd_pred_model_id}/{args.wcd_pred_model_id}_model.pt"
        
        # Check if the model file exists
        if not os.path.exists(model_label):
            raise FileNotFoundError(f"The model file does not exist at the path: {model_label}")
        
        # Load the model
        model = torch.load(model_label, map_location=torch.device(COMPUTE_DEVICE))
        model = model.eval()

    ratio = args.ratio
    
    if  "BOTH_UNIFORM" in experiment_type:
        blocking_rat = 1.05
        unblocking_rat = 1
    else:
        
        blocking_rat = int(ratio[0])
        unblocking_rat = int(ratio[2])
    model_id_folder =""
    if "PRED_WCD" in args.experiment_type:
        exp_folder = "ml-greedy"
        model_id_folder =f"{args.wcd_pred_model_id}"
    else:
        exp_folder = "greedy"
    
    data_storage_path = f"../wcd_optim_results/{exp_folder}/grid{grid_size}/timeout_{timeout_seconds}/{model_id_folder}/{experiment_type}/"
    create_folder(data_storage_path)
    
    if  "ALL_MODS" in experiment_type: #ALL Modifications are allowed
        ratio = f"ratio_{blocking_rat}_{unblocking_rat}"
        data_storage_path = f"{data_storage_path}/{ratio}/"
        create_folder(data_storage_path)
    
    create_folder(data_storage_path+"envs")
    create_folder(data_storage_path+"env_modifications")
    
    
    budgets = [np.round([(blocking_rat*max_budget)/(blocking_rat+unblocking_rat),(unblocking_rat*max_budget)/(blocking_rat+unblocking_rat) ]).tolist() for max_budget in total_budgets] #ratio of 1:1 blocking:unblocking
    
    
    print("Running experiment", experiment_type, "timeout",timeout_seconds)
    
    with open(f"../data/grid{grid_size}/model_training/dataset_{grid_size}{_label}.pkl", "rb") as f:
        loaded_dataset = pickle.load(f)
    
        
    if experiment_type == "BLOCKING_ONLY_EXHAUSTIVE":
        run_blocking_only_baseline(grid_size = grid_size, dataset = loaded_dataset, experiment_type =experiment_type, 
                                   use_exhaustive_search = True, verbose = False,env_run_index_seq=env_run_index_seq,timeout_seconds=timeout_seconds,
                                   budgets=total_budgets,data_storage_path=data_storage_path)
    elif experiment_type == "BLOCKING_ONLY_PRUNE_REDUCE":
        run_blocking_only_baseline(grid_size = grid_size, dataset = loaded_dataset, experiment_type =experiment_type, 
                                   use_exhaustive_search = False, verbose = False,env_run_index_seq=env_run_index_seq, timeout_seconds=timeout_seconds,
                                   budgets=total_budgets,data_storage_path=data_storage_path)
    elif experiment_type == "BLOCKING_ONLY_GREEDY_TRUE_WCD":
        run_blocking_only_greedy_true_wcd(grid_size = grid_size, dataset = loaded_dataset, experiment_type = experiment_type, 
                                          verbose = True,env_run_index_seq=env_run_index_seq,timeout_seconds = timeout_seconds,
                                          budgets=total_budgets,data_storage_path=data_storage_path)
    elif experiment_type == "BLOCKING_ONLY_GREEDY_PRED_WCD":
        run_blocking_only_greedy_pred_wcd(grid_size = grid_size, dataset = loaded_dataset, experiment_type = experiment_type,
                                                       prediction_model =model, verbose = False,env_run_index_seq=env_run_index_seq, 
                                          timeout_seconds=timeout_seconds, budgets=total_budgets,data_storage_path=data_storage_path)
    elif experiment_type in ["ALL_MODS_EXHAUSTIVE","BOTH_UNIFORM_EXHAUSTIVE"] :
        
        all_mods_uniform = True if experiment_type=="BOTH_UNIFORM_EXHAUSTIVE" else False
        
        run_all_modifications_exhuastive(grid_size = grid_size, dataset = loaded_dataset, experiment_type = experiment_type,
                                         env_run_index_seq=env_run_index_seq, timeout_seconds=timeout_seconds,budgets=budgets,
                                         data_storage_path=data_storage_path,all_mods_uniform=all_mods_uniform)
    elif experiment_type in ["ALL_MODS_GREEDY_TRUE_WCD","BOTH_UNIFORM_GREEDY_TRUE_WCD"]:
        
        all_mods_uniform = True if experiment_type=="BOTH_UNIFORM_GREEDY_TRUE_WCD" else False
        
        run_all_modifications_greedy_baseline_true_wcd(grid_size = grid_size, dataset = loaded_dataset, experiment_type = experiment_type,
                                                       env_run_index_seq=env_run_index_seq, timeout_seconds=timeout_seconds, budgets=budgets,
                                                       data_storage_path=data_storage_path,all_mods_uniform=all_mods_uniform)
    elif experiment_type in ["ALL_MODS_GREEDY_PRED_WCD","BOTH_UNIFORM_GREEDY_PRED_WCD"]:
        all_mods_uniform = True if experiment_type=="BOTH_UNIFORM_GREEDY_PRED_WCD" else False
        run_all_modifications_greedy_baseline_pred_wcd(grid_size = grid_size, dataset = loaded_dataset, experiment_type = experiment_type,
                                                       prediction_model =model, verbose = False,env_run_index_seq=env_run_index_seq, 
                                                       timeout_seconds=timeout_seconds, 
                                                       budgets=budgets,data_storage_path=data_storage_path, all_mods_uniform=all_mods_uniform)
    else:
        print ("INVALID EXPERIMENT CHOICE ")
    