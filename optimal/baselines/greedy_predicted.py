"""Converted from Greedy Predicted.ipynb.
Generated automatically by tools/notebook_to_script.py.
"""

import sys
import os
from pathlib import Path
from utils import *
from baseline_utils import *
from collections import deque
import time
import seaborn as sns
import multiprocessing
import pickle
from func_timeout import func_timeout, FunctionTimedOut

# %% [markdown]
# # GRD model
# - initial grid
# - optimal paths to goals
# - actions (positions that are visited on any of the paths)

# %% [code] cell 1

sys.path.insert(0, "../")

# %% [code] cell 2
def run_search_greedy_pred_wcd(root_grd_model, prediction_model):
    # If you have multiple GPUs, replace '0' with the appropriate GPU index
    device = 'cpu'

    # Set the device
    # torch.cuda.set_device(device)

    # Move the model and any data to the correct device
    root_grd_model
    prediction_model.to()

    start_time = time.time()

    # Make sure any data passed to the model is on the correct device
    final_grd = greedy_search_predicted_wcd(root_grd_model, model=prediction_model)

    end_time = time.time()

    time_taken = end_time - start_time
    wcd_diff = root_grd_model.get_wcd() - final_grd.get_wcd() if final_grd is not None else None

    # Move the result to CPU before putting it in the output queue
    return final_grd if final_grd is not None else None, time_taken, wcd_diff


def run_greedy_search_with_timeout(timeout, root_grd_model, prediction_model): # this is a special implementation due to issues with multithreading and CUDA
    try:
        # Run the function with a specified timeout
        result = func_timeout(timeout, run_search_greedy_pred_wcd, args=(root_grd_model, prediction_model))
        return result
    except FunctionTimedOut:
        # Handle the timeout case
        print(f"The function exceeded the {timeout} second timeout.")
        return None, 20, None

# %% [code] cell 5
with open(f"../data/dataset_6.pkl", "rb") as f:
        loaded_dataset = pickle.load(f)
grid_size =6
device ="cuda:0"
experiment_label = "ALL_MODS_GREEDY_PRED_WCD" #"baseline"

model_label = f"../models/wcd_nn_model.pt"
device ="cuda:0"
model = torch.load(model_label)
prediction_model = model.cpu().eval()

times = []
wcd_change = []


for i in range(0, len(loaded_dataset),10):
    x, y = loaded_dataset[i]  # Get a specific data sample
    x = x.unsqueeze(0).float().cuda()
    grid = decode_grid_design(x[0].cpu(), return_map=True)
    grid_size, goal_positions, blocked_positions, start_pos,unblocked_positions = decode_grid_design(x[0].cpu())

    is_valid, cost_to_goals = is_design_valid(grid_size, goal_positions, blocked_positions, start_pos)
    if not is_valid:
        print("INVALID Original evironment")

    wcd,paths,wcd_paths = compute_wcd_single_env(grid_size, goal_positions, blocked_positions, start_pos, vis_paths = False, return_paths = True)

    root_grd_model = GRDModel( grid_size = grid_size, start_pos = start_pos, goal_positions = goal_positions,
                              blocked_positions = blocked_positions, unblocked_positions = unblocked_positions,
                             init_goal_costs = cost_to_goals, compute_wcd = False)


    final_grd, time_taken, wcd_diff = run_with_timeout(20, root_grd_model, prediction_model)
    times.append(time_taken)
    wcd_change.append(wcd_diff)
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
    x_final = encode_from_grid_to_x(grid)

    # update_or_create_dataset(f"initial_envs_{grid_size}_{experiment_label}.pkl", [x], [y.item()]) # store the initial environments
    # update_or_create_dataset(f"final_envs_{grid_size}_{experiment_label}.pkl", [x_final], [final_grd.get_wcd()]) # store the final environments
    # create_or_update_list_file(f"data/times_{experiment_label}.csv",times)


    # Process your results here
    # print(f"Time taken: {time_taken} seconds")
    if final_grd:
        # Additional processing if final_grd is not None
        pass

    if i % 100 ==0:
        print(final_grd.get_wcd(),i, times[-1])

# %% [code] cell 6
run_search(root_grd_model, output,prediction_model)
