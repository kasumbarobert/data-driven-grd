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



def run_search_all_modifications(root_grd_model,max_budget, all_mods_uniform, output ):
    start_time = time.time()
    final_grd = breadth_first_search_all_actions(root_grd_model, max_budget, all_mods_uniform=all_mods_uniform)
    end_time = time.time()

    time_taken = end_time - start_time
    wcd_diff = root_grd_model.get_wcd() - final_grd.get_wcd() if final_grd is not None else None

    output.put((final_grd, time_taken, wcd_diff))

def run_search_greedy(root_grd_model,max_budget, all_mods_uniform, output):
    start_time = time.time()
    final_grd = greedy_search_true_wcd_all_mods(root_grd_model,max_budget,all_mods_uniform=all_mods_uniform)
    end_time = time.time()

    time_taken = end_time - start_time
    wcd_diff = root_grd_model.get_wcd() - final_grd.get_wcd() if final_grd is not None else None

    output.put((final_grd, time_taken, wcd_diff))
    
def run_search_blocking(root_grd_model, max_budget,output,use_exhaustive_search = True):
    start_time = time.time()
    final_grd = breadth_first_search_blocking(root_grd_model,max_budget,use_exhaustive_search)
    end_time = time.time()

    time_taken = end_time - start_time
    wcd_diff = root_grd_model.get_wcd() - final_grd.get_wcd() if final_grd is not None else None

    output.put((final_grd, time_taken, wcd_diff))

def run_search_greedy_true_wcd_blocking(root_grd_model,max_budget, output):
    start_time = time.time()
    final_grd = greedy_search_true_wcd_blocking_only(root_grd_model,max_budget)
    end_time = time.time()

    time_taken = end_time - start_time
    wcd_diff = root_grd_model.get_wcd() - final_grd.get_wcd() if final_grd is not None else None

    output.put((final_grd, time_taken, wcd_diff))
    
def run_search_greedy_pred_wcd_all_mods(root_grd_model,max_budget, prediction_model,all_mods_uniform=False):
    # If you have multiple GPUs, replace '0' with the appropriate GPU index
    device = 'cpu'

    # Set the device
    # torch.cuda.set_device(device)

    # Move the model and any data to the correct device
    root_grd_model
    prediction_model.to()

    start_time = time.time()

    # Make sure any data passed to the model is on the correct device
    final_grd = greedy_search_predicted_wcd_all_mods(root_grd_model, prediction_model,max_budget,all_mods_uniform=all_mods_uniform)

    end_time = time.time()

    time_taken = end_time - start_time
    wcd_diff = root_grd_model.get_wcd() - final_grd.get_wcd() if final_grd is not None else None

    # Move the result to CPU before putting it in the output queue
    return final_grd if final_grd is not None else None, time_taken, wcd_diff

def run_greedy_search_with_timeout_all_mods(timeout, root_grd_model, max_budget,prediction_model,all_mods_uniform=False): # this is a special implementation due to issues with multithreading and CUDA
    try:
        # Run the function with a specified timeout
        result = func_timeout(timeout, run_search_greedy_pred_wcd_all_mods, args=(root_grd_model,max_budget,prediction_model,all_mods_uniform))
        return result
    except FunctionTimedOut:
        # Handle the timeout case
        print(f"The function exceeded the {timeout} second timeout.")
        return None, timeout, None
    

def run_search_greedy_pred_wcd_blocking_only(root_grd_model,max_budget, prediction_model):
    # If you have multiple GPUs, replace '0' with the appropriate GPU index
    device = 'cpu'

    # Set the device
    # torch.cuda.set_device(device)

    # Move the model and any data to the correct device
    root_grd_model
    prediction_model.to()

    start_time = time.time()

    # Make sure any data passed to the model is on the correct device
    final_grd = greedy_search_pred_wcd_blocking_only(root_grd_model, max_budget,prediction_model)

    end_time = time.time()

    time_taken = end_time - start_time

    # Move the result to CPU before putting it in the output queue
    return final_grd if final_grd is not None else None, time_taken, None

def run_greedy_search_with_timeout_blocking_only(timeout, root_grd_model, max_budget,prediction_model): # this is a special implementation due to issues with multithreading and CUDA
    try:
        # Run the function with a specified timeout
        result = func_timeout(timeout, run_search_greedy_pred_wcd_blocking_only, args=(root_grd_model,max_budget,prediction_model))
        return result
    except FunctionTimedOut:
        # Handle the timeout case
        print(f"The function exceeded the {timeout} second timeout.")
        return None, timeout, None


    
def run_all_modifications_exhuastive(grid_size = 6, dataset = None, experiment = "debug", verbose = False,num_instances=200,
                                     timeout_seconds = 20,budgets=[1,2], data_storage_path=None,
                                    all_mods_uniform=False):
    times = []
    wcd_change = []
    num_changes =[]
    for i in range(0, len(dataset),len(dataset)//num_instances):
        x, y = loaded_dataset[i]  # Get a specific data sample
        x = x.unsqueeze(0).float().to(DEVICE)
        grid = decode_grid_design(x[0].cpu(), return_map=True)
        grid_size, goal_positions, blocked_positions, start_pos,unblocked_positions = decode_grid_design(x[0].cpu())

        is_valid, cost_to_goals = is_design_valid(grid_size, goal_positions, blocked_positions, start_pos)
        if not is_valid:
            print("INVALID Original evironment")

        wcd,paths,wcd_paths = compute_wcd_single_env(grid_size, goal_positions, blocked_positions, start_pos, vis_paths = False, return_paths = True)

        budget_times = []
        budget_wcd_change = []
        budget_num_changes =[]
        for max_budget in budgets:
            root_grd_model = GRDModel( grid_size = grid_size, start_pos = start_pos, goal_positions = goal_positions,
                                  blocked_positions = blocked_positions, unblocked_positions = unblocked_positions,
                                 init_goal_costs = cost_to_goals, n_max_changes =max_budget )
            # Create a queue to hold the output
            output = multiprocessing.Queue()

            # Create and start the process
            search_process = multiprocessing.Process(target=run_search_all_modifications, args=(root_grd_model,
                                                                                                max_budget,all_mods_uniform, output))
            search_process.start()


            # Wait for the specified timeout or until the process completes
            search_process.join(timeout=timeout_seconds)

            if search_process.is_alive():
                # Terminate the process if it is still running after the timeout
                search_process.terminate()
                search_process.join()
                print("Search was terminated due to timeout.")
                time_taken = timeout_seconds  # Record the timeout duration as the time taken
                final_grd = None
                wcd_diff = None
            else:
                # Process finished within timeout, retrieve the result
                final_grd, time_taken, wcd_diff = output.get()
                # print(final_grd.get_wcd())
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
            
            budget_wcd_change.append(y.item()-final_grd.get_wcd())
            
            x_changes = x_final.cpu()[:, 1, :, :]-x.cpu()[:, 1, :, :]
            
            blockings = (x_changes==1).sum(axis=(1, 2)).item()
            removals = (x_changes==-1).sum(axis=(1, 2)).item()
            
            budget_num_changes.append([blockings,removals])
            
            if time_taken == timeout_seconds: # no need to run higher budgets if smaller ones timeout
                target_length = len(budgets)
                # Filling up 'budget_times' and 'wcd_change' with their last value to match the length of 'budgets'
                budget_times.extend([budget_times[-1]] * (target_length - len(budget_times)))
                budget_wcd_change.extend([budget_wcd_change[-1]] * (target_length - len(budget_wcd_change)))
                budget_num_changes.extend([budget_num_changes[-1]] * (target_length - len(budget_num_changes)))
                break

        times.append(budget_times)
        wcd_change.append(budget_wcd_change)
        num_changes.append(budget_num_changes)
        
        update_or_create_dataset(f"{data_storage_path}/initial_envs_{grid_size}_{experiment}.pkl", [x], [y.item()]) # store the initial environments
        update_or_create_dataset(f"{data_storage_path}/final_envs_{grid_size}_{experiment}.pkl", [x_final], [final_grd.get_wcd()]) # store the final environments
        create_or_update_list_file(f"{data_storage_path}/times_{grid_size}_{experiment}.csv",times)
        create_or_update_list_file(f"{data_storage_path}/wcd_change_{grid_size}_{experiment}.csv",wcd_change)
        create_or_update_list_file(f"{data_storage_path}/budgets_{grid_size}_{experiment}.csv",[np.sum(budgets,axis =1)])
        create_or_update_list_file(f"{data_storage_path}/num_changes_{grid_size}_{experiment}.csv",num_changes)


        if i % 100 ==0 and verbose:
            print(final_grd.get_wcd(),i, times[-1])
    

def run_all_modifications_greedy_baseline_true_wcd(grid_size = 6, dataset = None, experiment = "debug", verbose = False,
                                                   num_instances=200,timeout_seconds = 20, budgets =[1,10],data_storage_path=None,
                                                  all_mods_uniform=False):

    times = []
    wcd_change = []
    num_changes = []
    num_changes =[]
    for i in range(0, len(dataset),len(dataset)//num_instances):
        x, y = loaded_dataset[i]  # Get a specific data sample
        x = x.unsqueeze(0).float().to(DEVICE)
        grid = decode_grid_design(x[0].cpu(), return_map=True)
        grid_size, goal_positions, blocked_positions, start_pos,unblocked_positions = decode_grid_design(x[0].cpu())

        is_valid, cost_to_goals = is_design_valid(grid_size, goal_positions, blocked_positions, start_pos)
        if not is_valid:
            print("INVALID Original evironment")
        wcd,paths,wcd_paths = compute_wcd_single_env(grid_size, goal_positions, blocked_positions, start_pos, vis_paths = False, return_paths = True)

        budget_times = []
        budget_wcd_change = []
        budget_num_changes =[]
        for max_budget in budgets:
            print("MAX Budget is",max_budget,"ALL_MODS",all_mods_uniform)
            
            root_grd_model = GRDModel( grid_size = grid_size, start_pos = start_pos, goal_positions = goal_positions,
                                  blocked_positions = blocked_positions, unblocked_positions = unblocked_positions,
                                 init_goal_costs = cost_to_goals,n_max_changes =max_budget ) 
            # Create a queue to hold the output
            output = multiprocessing.Queue()

            # Create and start the process
            search_process = multiprocessing.Process(target=run_search_greedy, args=(root_grd_model,max_budget,all_mods_uniform, output))
            search_process.start()


            # Wait for the specified timeout or until the process completes
            search_process.join(timeout=timeout_seconds)

            if search_process.is_alive():
                # Terminate the process if it is still running after the timeout
                search_process.terminate()
                search_process.join()
                print("Search was terminated due to timeout.")
                time_taken = timeout_seconds  # Record the timeout duration as the time taken
                final_grd = None
                wcd_diff = None
            else:
                # Process finished within timeout, retrieve the result
                final_grd, time_taken, wcd_diff = output.get()
            
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
        
            budget_wcd_change.append(y.item()-final_grd.get_wcd())
            
            x_changes = x_final.cpu()[:, 1, :, :]-x.cpu()[:, 1, :, :]
            blockings = (x_changes==1).sum(axis=(1, 2)).item()
            removals = (x_changes==-1).sum(axis=(1, 2)).item()
            
            budget_num_changes.append([blockings,removals])
            
            if time_taken == timeout_seconds: # no need to run higher budgets if smaller ones timeout
                target_length = len(budgets)
                # Filling up 'budget_times' and 'wcd_change' with their last value to match the length of 'budgets'
                budget_times.extend([budget_times[-1]] * (target_length - len(budget_times)))
                budget_wcd_change.extend([budget_wcd_change[-1]] * (target_length - len(budget_wcd_change)))
                budget_num_changes.extend([budget_num_changes[-1]] * (target_length - len(budget_num_changes)))
                break
                
            update_or_create_dataset(f"{data_storage_path}/initial_envs_{grid_size}_{experiment}.pkl", [x], [y.item()]) # store the initial environments
            update_or_create_dataset(f"{data_storage_path}/final_envs_{grid_size}_{experiment}.pkl", [x_final], [final_grd.get_wcd()]) # store the final environments

        times.append(budget_times)
        wcd_change.append(budget_wcd_change)
        num_changes.append(budget_num_changes)
        
        create_or_update_list_file(f"{data_storage_path}/times_{grid_size}_{experiment}.csv",times)
        create_or_update_list_file(f"{data_storage_path}/wcd_change_{grid_size}_{experiment}.csv",wcd_change)
        create_or_update_list_file(f"{data_storage_path}/budgets_{grid_size}_{experiment}.csv",[np.sum(budgets,axis =1)])
        create_or_update_list_file(f"{data_storage_path}/num_changes_{grid_size}_{experiment}.csv",num_changes)

        if i % 100 ==0 and verbose:
            print(final_grd.get_wcd(),i, times[-1])


def run_all_modifications_greedy_baseline_pred_wcd(grid_size = 6, dataset = None, experiment = "debug", 
                                                   verbose = False, prediction_model = None, num_instances=200,timeout_seconds = 20, 
                                                   budgets = [1,2], data_storage_path=None, all_mods_uniform=False):
    times = []
    wcd_change = []
    num_changes =[]
    for i in range(0, len(dataset),len(dataset)//num_instances):
        x, y = loaded_dataset[i]  # Get a specific data sample
        x = x.unsqueeze(0).float().to(DEVICE)
        grid = decode_grid_design(x[0].cpu(), return_map=True)
        grid_size, goal_positions, blocked_positions, start_pos,unblocked_positions = decode_grid_design(x[0].cpu())

        is_valid, cost_to_goals = is_design_valid(grid_size, goal_positions, blocked_positions, start_pos)
        if not is_valid:
            print("INVALID Original evironment")

        wcd,paths,wcd_paths = compute_wcd_single_env(grid_size, goal_positions, blocked_positions, start_pos, vis_paths = False, return_paths = True)

        budget_times = []
        budget_wcd_change = []
        budget_num_changes =[]
        for max_budget in budgets:
            root_grd_model = GRDModel( grid_size = grid_size, start_pos = start_pos, goal_positions = goal_positions,
                          blocked_positions = blocked_positions, unblocked_positions = unblocked_positions,
                         init_goal_costs = cost_to_goals, compute_wcd = False, n_max_changes =max_budget )
            final_grd, time_taken, wcd_diff= run_greedy_search_with_timeout_all_mods(timeout_seconds, root_grd_model,max_budget, prediction_model,all_mods_uniform=all_mods_uniform)
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
            budget_wcd_change.append(y.item()-final_grd.get_wcd())
            x_changes = x_final.cpu()[:, 1, :, :]-x.cpu()[:, 1, :, :]
            blockings = (x_changes==1).sum(axis=(1, 2)).item()
            removals = (x_changes==-1).sum(axis=(1, 2)).item()
            
            budget_num_changes.append([blockings,removals])
            
            if time_taken == timeout_seconds: # no need to run higher budgets if smaller ones timeout
                target_length = len(budgets)
                # Filling up 'budget_times' and 'wcd_change' with their last value to match the length of 'budgets'
                budget_times.extend([budget_times[-1]] * (target_length - len(budget_times)))
                budget_wcd_change.extend([budget_wcd_change[-1]] * (target_length - len(budget_wcd_change)))
                budget_num_changes.extend([budget_num_changes[-1]] * (target_length - len(budget_num_changes)))
                break
                
            update_or_create_dataset(f"{data_storage_path}/initial_envs_{grid_size}_{experiment}.pkl", [x], [y.item()]) # store the initial environments
            update_or_create_dataset(f"{data_storage_path}/final_envs_{grid_size}_{experiment}.pkl", [x_final], [final_grd.get_wcd()]) # store the final environments

        times.append(budget_times)
        wcd_change.append(budget_wcd_change)
        num_changes.append(budget_num_changes)
        
        create_or_update_list_file(f"{data_storage_path}/times_{grid_size}_{experiment}.csv",times)
        create_or_update_list_file(f"{data_storage_path}/wcd_change_{grid_size}_{experiment}.csv",wcd_change)
        create_or_update_list_file(f"{data_storage_path}/budgets_{grid_size}_{experiment}.csv",[np.sum(budgets,axis =1)])
        create_or_update_list_file(f"{data_storage_path}/num_changes_{grid_size}_{experiment}.csv",num_changes)
        

        if i % 100 ==0 and verbose:
            print(final_grd.get_wcd(),i, times[-1])

def run_blocking_only_baseline(grid_size = 6, dataset = None, experiment = "debug", use_exhaustive_search = False, verbose = False,
                               num_instances=200,timeout_seconds = 20,budgets=[1,2],data_storage_path=None):
    times = []
    wcd_change = []
    num_changes =[]
    for i in range(0, len(dataset),len(dataset)//num_instances):
        x, y = loaded_dataset[i]  # Get a specific data sample
        x = x.unsqueeze(0).float().to(DEVICE)
        grid = decode_grid_design(x[0].cpu(), return_map=True)
        grid_size, goal_positions, blocked_positions, start_pos,space_pos = decode_grid_design(x[0].cpu())

        wcd,paths,wcd_paths = compute_wcd_single_env(grid_size, goal_positions, blocked_positions, start_pos, vis_paths = False, return_paths = True)

        root_grd_model = GRDModelBlockingOnly( grid, paths,blocked = [],fixed_positions = (grid_size, goal_positions, start_pos))
        
        budget_times = []
        budget_wcd_change = []
        budget_num_changes =[]
        for max_budget in budgets:
            # Create a queue to hold the output
            output = multiprocessing.Queue()

            # Create and start the process
            search_process = multiprocessing.Process(target=run_search_blocking, args=(root_grd_model,max_budget, output,use_exhaustive_search))
            search_process.start()


            # Wait for the specified timeout or until the process completes
            search_process.join(timeout=timeout_seconds)

            if search_process.is_alive():
                # Terminate the process if it is still running after the timeout
                search_process.terminate()
                search_process.join()
                print("Search was terminated due to timeout.")
                time_taken = timeout_seconds  # Record the timeout duration as the time taken
                final_grd = None
                wcd_diff = None
            else:
                # Process finished within timeout, retrieve the result
                final_grd, time_taken, wcd_diff = output.get()

            if not final_grd is None:
                grid = final_grd.get_grid()
                for b in final_grd.blocked:
                    grid[b[0],b[1]] ="X"
            else:
                grid = root_grd_model.grid
                final_grd = root_grd_model
                
            x_final = encode_from_grid_to_x(grid)
            
            budget_times.append(time_taken)
            budget_wcd_change.append(y.item()-final_grd.get_wcd())
            
            x_changes = x_final.cpu()[:, 1, :, :]-x.cpu()[:, 1, :, :]
            blockings = (x_changes==1).sum(axis=(1, 2)).item()
            removals = (x_changes==-1).sum(axis=(1, 2)).item()
            
            budget_num_changes.append([blockings,removals])
            
            if time_taken == timeout_seconds: # no need to run higher budgets if smaller ones timeout
                target_length = len(budgets)
                # Filling up 'budget_times' and 'wcd_change' with their last value to match the length of 'budgets'
                budget_times.extend([budget_times[-1]] * (target_length - len(budget_times)))
                budget_wcd_change.extend([budget_wcd_change[-1]] * (target_length - len(budget_wcd_change)))
                budget_num_changes.extend([budget_num_changes[-1]] * (target_length - len(budget_num_changes)))
                break
                
            update_or_create_dataset(f"{data_storage_path}/initial_envs_{grid_size}_{experiment}.pkl", [x], [y.item()]) # store the initial environments
            update_or_create_dataset(f"{data_storage_path}/final_envs_{grid_size}_{experiment}.pkl", [x_final], [final_grd.get_wcd()]) # store the final environments

        times.append(budget_times)
        wcd_change.append(budget_wcd_change)
        num_changes.append(budget_num_changes)
        
        create_or_update_list_file(f"{data_storage_path}/times_{grid_size}_{experiment}.csv",times)
        create_or_update_list_file(f"{data_storage_path}/wcd_change_{grid_size}_{experiment}.csv",wcd_change)
        create_or_update_list_file(f"{data_storage_path}/budgets_{grid_size}_{experiment}.csv",[budgets])
        create_or_update_list_file(f"{data_storage_path}/num_changes_{grid_size}_{experiment}.csv",num_changes)

        if i % 100 ==0 and verbose:
            print(i, times[-1])

def run_blocking_only_greedy_true_wcd(grid_size = 6, dataset = None, experiment = "debug", verbose = False,
                                      num_instances=200,timeout_seconds = 20, budgets =[1,2],data_storage_path=None):
    times = []
    wcd_change = []
    num_changes =[]

    for i in range(0, len(loaded_dataset),len(loaded_dataset)//num_instances):
        x, y = loaded_dataset[i]  # Get a specific data sample
        x = x.unsqueeze(0).float().to(DEVICE)
        grid = decode_grid_design(x[0].cpu(), return_map=True)
        grid_size, goal_positions, blocked_positions, start_pos,space_pos = decode_grid_design(x[0].cpu())

        wcd,paths,wcd_paths = compute_wcd_single_env(grid_size, goal_positions, blocked_positions, start_pos, vis_paths = False, return_paths = True)

        root_grd_model = GRDModelBlockingOnly( grid, paths,blocked = [], fixed_positions = (grid_size, goal_positions, start_pos))

        budget_times = []
        budget_wcd_change = []
        budget_num_changes =[]
        for max_budget in budgets:
            # Create a queue to hold the output
            output = multiprocessing.Queue()

            # Create and start the process
            search_process = multiprocessing.Process(target=run_search_greedy_true_wcd_blocking, args=(root_grd_model,max_budget, output))
            search_process.start()


            # Wait for the specified timeout or until the process completes
            search_process.join(timeout=timeout_seconds)

            if search_process.is_alive():
                # Terminate the process if it is still running after the timeout
                search_process.terminate()
                search_process.join()
                print("Search was terminated due to timeout.")
                time_taken = timeout_seconds  # Record the timeout duration as the time taken
                final_grd = None
                wcd_diff = None
            else:
                # Process finished within timeout, retrieve the result
                final_grd, time_taken, wcd_diff = output.get()

            if not final_grd is None:
                grid = final_grd.get_grid()
                for b in final_grd.blocked:
                    grid[b[0],b[1]] ="X"
            else:
                grid = root_grd_model.grid
                final_grd = root_grd_model
            x_final = encode_from_grid_to_x(grid)
            
            budget_times.append(time_taken)
            budget_wcd_change.append(y.item()-final_grd.get_wcd())
            
            x_changes = x_final.cpu()[:, 1, :, :]-x.cpu()[:, 1, :, :]
            blockings = (x_changes==1).sum(axis=(1, 2)).item()
            removals = (x_changes==-1).sum(axis=(1, 2)).item()
            
            budget_num_changes.append([blockings,removals])
            
            if time_taken == timeout_seconds: # no need to run higher budgets if smaller ones timeout
                target_length = len(budgets)
                # Filling up 'budget_times' and 'wcd_change' with their last value to match the length of 'budgets'
                budget_times.extend([budget_times[-1]] * (target_length - len(budget_times)))
                budget_wcd_change.extend([budget_wcd_change[-1]] * (target_length - len(budget_wcd_change)))
                budget_num_changes.extend([budget_num_changes[-1]] * (target_length - len(budget_num_changes)))
                break
                
            update_or_create_dataset(f"{data_storage_path}/initial_envs_{grid_size}_{experiment}.pkl", [x], [y.item()]) # store the initial environments
            update_or_create_dataset(f"{data_storage_path}/final_envs_{grid_size}_{experiment}.pkl", [x_final], [final_grd.get_wcd()]) # store the final environments

        times.append(budget_times)
        wcd_change.append(budget_wcd_change)
        num_changes.append(budget_num_changes)
        
        create_or_update_list_file(f"{data_storage_path}/times_{grid_size}_{experiment}.csv",times)
        create_or_update_list_file(f"{data_storage_path}/wcd_change_{grid_size}_{experiment}.csv",wcd_change)
        create_or_update_list_file(f"{data_storage_path}/budgets_{grid_size}_{experiment}.csv",[budgets])
        create_or_update_list_file(f"{data_storage_path}/num_changes_{grid_size}_{experiment}.csv",num_changes)

        if i % 100 ==0 and verbose:
            print(i, times[-1])

def run_blocking_only_greedy_pred_wcd(grid_size = 6, dataset = None, experiment = "debug", verbose = False, 
                                      prediction_model = None, num_instances=200,timeout_seconds = 20, budgets =[1,2],data_storage_path=None):
    times = []
    wcd_change = []
    num_changes =[]

    for i in range(0, len(loaded_dataset),len(loaded_dataset)//num_instances):
        x, y = loaded_dataset[i]  # Get a specific data sample
        x = x.unsqueeze(0).float().to(DEVICE)
        grid = decode_grid_design(x[0].cpu(), return_map=True)
        grid_size, goal_positions, blocked_positions, start_pos,space_pos = decode_grid_design(x[0].cpu())

        wcd,paths,wcd_paths = compute_wcd_single_env(grid_size, goal_positions, blocked_positions, start_pos, vis_paths = False, return_paths = True)

        root_grd_model = GRDModelBlockingOnly( grid, paths,blocked = [], compute_wcd = False,fixed_positions = (grid_size, goal_positions, start_pos))

        budget_times = []
        budget_wcd_change = []
        budget_num_changes =[]
        for max_budget in budgets:
            final_grd, time_taken, wcd_diff= run_greedy_search_with_timeout_blocking_only(timeout_seconds, root_grd_model,max_budget, prediction_model)
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
            
            if time_taken == timeout_seconds: # no need to run higher budgets if smaller ones timeout
                target_length = len(budgets)
                # Filling up 'budget_times' and 'wcd_change' with their last value to match the length of 'budgets'
                budget_times.extend([budget_times[-1]] * (target_length - len(budget_times)))
                budget_wcd_change.extend([budget_wcd_change[-1]] * (target_length - len(budget_wcd_change)))
                budget_num_changes.extend([budget_num_changes[-1]] * (target_length - len(budget_num_changes)))
                break
                
            update_or_create_dataset(f"{data_storage_path}/initial_envs_{grid_size}_{experiment}.pkl", [x], [y.item()]) # store the initial environments
            update_or_create_dataset(f"{data_storage_path}/final_envs_{grid_size}_{experiment}.pkl", [x_final], [final_wcd]) # store the final environments

        times.append(budget_times)
        wcd_change.append(budget_wcd_change)
        num_changes.append(budget_num_changes)
        
        create_or_update_list_file(f"{data_storage_path}/times_{grid_size}_{experiment}.csv",times)
        create_or_update_list_file(f"{data_storage_path}/wcd_change_{grid_size}_{experiment}.csv",wcd_change)
        create_or_update_list_file(f"{data_storage_path}/budgets_{grid_size}_{experiment}.csv",[budgets])
        create_or_update_list_file(f"{data_storage_path}/num_changes_{grid_size}_{experiment}.csv",num_changes)

        if i % 100 ==0 and verbose:
            print(i, times[-1])

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Baseline Experiments for Gridworld - Optimal agent")
    
    parser.add_argument(
        "--experiment",
        type=str,  # Ensure that the input is expected to be a string
        default="BLOCKING_ONLY_GREEDY_PRED_WCD",  # Set a default label for the experiment
        choices =["BLOCKING_ONLY_EXHUASTIVE", "BLOCKING_ONLY_PRUNE_REDUCE","BLOCKING_ONLY_GREEDY_TRUE_WCD","BLOCKING_ONLY_GREEDY_PRED_WCD","BOTH_UNIFORM_EXHAUSTIVE",
                  "BOTH_UNIFORM_GREEDY_PRED_WCD","BOTH_UNIFORM_GREEDY_TRUE_WCD"],
        # choices =["ALL_MODS_EXHUASTIVE","ALL_MODS_GREEDY_TRUE_WCD","ALL_MODS_GREEDY_PRED_WCD"],
        help="Label for the current experiment run. Default is 'BLOCKING_ONLY_EXHUASTIVE'.",
    )
    parser.add_argument(
        "--grid_size",
        type=int,  # Ensure that the input is expected to be a int
        default=10,  # Set the default value to 1
        choices = [6,10], # 
        help="Maximum grid size.",
    )
    
    parser.add_argument(
        "--num_instances",
        type=int,  # Ensure that the input is expected to be a int
        default=500,  # Set the default value to 1
        help="spacing in the test dataset to use for the experiment",
    )
    
    parser.add_argument(
        "--timeout_seconds",
        type=int,  # Ensure that the input is expected to be a int
        default=600,  # Set the default value to 1
        help="Timeout seconds",
    )
    
    current_directory = os.path.dirname(os.path.realpath(__file__))
    
    args = parser.parse_args()
    experiment = args.experiment
    grid_size = args.grid_size
    num_instances = args.num_instances
    timeout_seconds = args.timeout_seconds

    total_budgets = [1,3,5,7,9,11,13,15,17,19]
    
    budgets = [np.round([(1*max_budget)/4,(3*max_budget)/4 ]).tolist() for max_budget in total_budgets] #ratio of 3:5 blocking:unblocking
    
    data_storage_path = f"{current_directory}/baselines/data/grid{grid_size}/timeout_{timeout_seconds}"
    create_folder(data_storage_path)
    
    
    
    print("Running experiment", experiment, "timeout",timeout_seconds, " using", DEVICE, "for", num_instances, "instances")
    print("After each instance, result files  are stored at",data_storage_path)
    
    with open(f"{current_directory}/data/grid{grid_size}/model_training/dataset_{grid_size}_best.pkl", "rb") as f:
        loaded_dataset = pickle.load(f)
    
        
    if experiment == "BLOCKING_ONLY_EXHUASTIVE":
        run_blocking_only_baseline(grid_size = grid_size, dataset = loaded_dataset, experiment =experiment, 
                                   use_exhaustive_search = True, verbose = False,num_instances=num_instances,timeout_seconds=timeout_seconds,
                                   budgets=total_budgets,data_storage_path=data_storage_path)
    elif experiment == "BLOCKING_ONLY_PRUNE_REDUCE":
        run_blocking_only_baseline(grid_size = grid_size, dataset = loaded_dataset, experiment =experiment, 
                                   use_exhaustive_search = False, verbose = False,num_instances=num_instances, timeout_seconds=timeout_seconds,
                                   budgets=total_budgets,data_storage_path=data_storage_path)
    elif experiment == "BLOCKING_ONLY_GREEDY_TRUE_WCD":
        run_blocking_only_greedy_true_wcd(grid_size = grid_size, dataset = loaded_dataset, experiment = experiment, 
                                          verbose = True,num_instances=num_instances,timeout_seconds = timeout_seconds,
                                          budgets=total_budgets,data_storage_path=data_storage_path)
    elif experiment == "BLOCKING_ONLY_GREEDY_PRED_WCD":
        model_label = f"{current_directory}/models/wcd_nn_model_{grid_size}_best.pt"
        device ="cuda:0"
        model = torch.load(model_label)
        model = model.to(DEVICE).eval()
        run_blocking_only_greedy_pred_wcd(grid_size = grid_size, dataset = loaded_dataset, experiment = experiment,
                                                       prediction_model =model, verbose = False,num_instances=num_instances, 
                                          timeout_seconds=timeout_seconds, budgets=total_budgets,data_storage_path=data_storage_path)
    elif experiment in ["ALL_MODS_EXHUASTIVE","BOTH_UNIFORM_EXHAUSTIVE"] :
        
        all_mods_uniform = True if experiment=="BOTH_UNIFORM_EXHAUSTIVE" else False
        
        run_all_modifications_exhuastive(grid_size = grid_size, dataset = loaded_dataset, experiment = experiment,
                                         num_instances=num_instances, timeout_seconds=timeout_seconds,budgets=budgets,
                                         data_storage_path=data_storage_path,all_mods_uniform=all_mods_uniform)
    elif experiment in ["ALL_MODS_GREEDY_TRUE_WCD","BOTH_UNIFORM_GREEDY_TRUE_WCD"]:
        
        all_mods_uniform = True if experiment=="BOTH_UNIFORM_GREEDY_TRUE_WCD" else False
        
        run_all_modifications_greedy_baseline_true_wcd(grid_size = grid_size, dataset = loaded_dataset, experiment = experiment,
                                                       num_instances=num_instances, timeout_seconds=timeout_seconds, budgets=budgets,
                                                       data_storage_path=data_storage_path,all_mods_uniform=all_mods_uniform)
    elif experiment in ["ALL_MODS_GREEDY_PRED_WCD","BOTH_UNIFORM_GREEDY_PRED_WCD"]:
        
        all_mods_uniform = True if experiment=="BOTH_UNIFORM_GREEDY_PRED_WCD" else False
        
        model_label = f"{current_directory}/models/wcd_nn_model_{grid_size}_best.pt"

        model = torch.load(model_label)
        model = model.to(DEVICE)
        run_all_modifications_greedy_baseline_pred_wcd(grid_size = grid_size, dataset = loaded_dataset, experiment = experiment,
                                                       prediction_model =model, verbose = False,num_instances=num_instances, 
                                                       timeout_seconds=timeout_seconds, 
                                                       budgets=budgets,data_storage_path=data_storage_path, all_mods_uniform=all_mods_uniform)
    else:
        print ("INVALID EXPERIMENT CHOICE ")
    