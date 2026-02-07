"""Converted from Blocking Actions Only Baselines.ipynb.
Generated automatically by tools/notebook_to_script.py.
"""

import numpy as np
import pickle
import sys
import os
from pathlib import Path
from utils import *
from baseline_utils import *
from collections import deque
import time
import seaborn as sns
import multiprocessing

# %% [markdown]
# # GRD model
# - initial grid
# - optimal paths to goals
# - actions (positions that are visited on any of the paths)

# %% [code] cell 1

sys.path.insert(0, "../")

# %% [code] cell 2
# np.flatten([(9,0),(7,8),[(9,0)]])

# %% [code] cell 4
def run_search_greedy_true_wcd_blocking_only(root_grd_model, output):
    start_time = time.time()
    final_grd = greedy_search_true_wcd_blocking_only(root_grd_model)
    end_time = time.time()

    time_taken = end_time - start_time
    wcd_diff = root_grd_model.get_wcd() - final_grd.get_wcd() if final_grd is not None else None

    output.put((final_grd, time_taken, wcd_diff))

# %% [code] cell 5
def run_blocking_only_greedy(grid_size = 6, dataset = None, experiment_label = "debug", verbose = False,interval=200,timeout_seconds = 20):
    times = []
    wcd_change = []
    # len(loaded_dataset)/

    for i in range(0, len(loaded_dataset),interval):
        x, y = loaded_dataset[i]  # Get a specific data sample
        x = x.unsqueeze(0).float().cuda()
        grid = decode_grid_design(x[0].cpu(), return_map=True)
        grid_size, goal_positions, blocked_positions, start_pos,space_pos = decode_grid_design(x[0].cpu())

        wcd,paths,wcd_paths = compute_wcd_single_env(grid_size, goal_positions, blocked_positions, start_pos, vis_paths = False, return_paths = True)

        root_grd_model = GRDModelBlockingOnly( grid, paths,blocked = [])

        # Create a queue to hold the output
        output = multiprocessing.Queue()

        # Create and start the process
        search_process = multiprocessing.Process(target=run_search_greedy_true_wcd_blocking, args=(root_grd_model, output))
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
            times.append(time_taken)
        else:
            # Process finished within timeout, retrieve the result
            final_grd, time_taken, wcd_diff = output.get()
            times.append(time_taken)
            wcd_change.append(wcd_diff)

        if not final_grd is None:
            grid = final_grd.get_grid()
            for b in final_grd.blocked:
                grid[b[0],b[1]] ="X"
        else:
            grid = root_grd_model.grid
            final_grd = root_grd_model
        x_final = encode_from_grid_to_x(grid)


        update_or_create_dataset(f"initial_envs_{grid_size}_{experiment_label}.pkl", [x], [y.item()]) # store the initial environments
        update_or_create_dataset(f"final_envs_{grid_size}_{experiment_label}.pkl", [x_final], [final_grd.get_wcd()]) # store the final environments
        create_or_update_list_file(f"data/times_{experiment_label}.csv",times)

        if i % 100 ==0 and verbose:
            print(i, times[-1])

# %% [code] cell 6
run_blocking_only_greedy(grid_size = 6, dataset = None, experiment_label = "debug", verbose = True,interval=200,timeout_seconds = 20)

# %% [code] cell 8
def run_search_blocking(root_grd_model, output,use_exhaustive_search = True):
    start_time = time.time()
    final_grd = breadth_first_search_blocking(root_grd_model,use_exhaustive_search)
    end_time = time.time()

    time_taken = end_time - start_time
    wcd_diff = root_grd_model.get_wcd() - final_grd.get_wcd() if final_grd is not None else None

    output.put((final_grd, time_taken, wcd_diff))

# %% [code] cell 9
with open(f"../data/dataset_6.pkl", "rb") as f:
        loaded_dataset = pickle.load(f)
grid_size =6
device ="cuda:0"
use_exhaustive_search = False
experiment_label = f'blocking_only_{"exhuastive" if use_exhaustive_search else "prune"}' #baseline

# %% [code] cell 10

def run_blocking_only_baseline(grid_size = 6, dataset = None, experiment_label = "debug", use_exhaustive_search = False, verbose = False):
    times = []
    wcd_change = []
    # len(loaded_dataset)/

    for i in range(0, len(loaded_dataset),200):
        x, y = loaded_dataset[i]  # Get a specific data sample
        x = x.unsqueeze(0).float().cuda()
        grid = decode_grid_design(x[0].cpu(), return_map=True)
        grid_size, goal_positions, blocked_positions, start_pos,space_pos = decode_grid_design(x[0].cpu())

        wcd,paths,wcd_paths = compute_wcd_single_env(grid_size, goal_positions, blocked_positions, start_pos, vis_paths = False, return_paths = True)

        root_grd_model = GRDModelBlockingOnly( grid, paths,blocked = [])

        # Create a queue to hold the output
        output = multiprocessing.Queue()

        # Create and start the process
        search_process = multiprocessing.Process(target=run_search_blocking, args=(root_grd_model, output,use_exhaustive_search))
        search_process.start()

        # Set your timeout duration
        timeout_seconds = 10  # 10 s

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
            times.append(time_taken)
        else:
            # Process finished within timeout, retrieve the result
            final_grd, time_taken, wcd_diff = output.get()
            times.append(time_taken)
            wcd_change.append(wcd_diff)

        if not final_grd is None:
            grid = final_grd.get_grid()
            for b in final_grd.blocked:
                grid[b[0],b[1]] ="X"
        else:
            grid = root_grd_model.grid
            final_grd = root_grd_model
        x_final = encode_from_grid_to_x(grid)


        update_or_create_dataset(f"initial_envs_{grid_size}_{experiment_label}.pkl", [x], [y.item()]) # store the initial environments
        update_or_create_dataset(f"final_envs_{grid_size}_{experiment_label}.pkl", [x_final], [final_grd.get_wcd()]) # store the final environments
        create_or_update_list_file(f"data/times_{experiment_label}.csv",times)

        if i % 100 ==0 and verbose:
            print(i, times[-1])

# %% [code] cell 11
run_blocking_only_baseline(grid_size = 6, dataset = loaded_dataset, experiment_label =experiment_label, use_exhaustive_search = True, verbose = True)
