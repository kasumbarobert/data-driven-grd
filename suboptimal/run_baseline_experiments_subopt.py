import sys
# sys.path.insert(0, "./")
# sys.path.insert(0, "../../")
import torch
from torch.utils.data import Dataset
from utils_suboptimal import *
import random
import matplotlib.pyplot as plt

from torchvision.models import resnet50, resnet18
import argparse
import traceback
import time
import seaborn as sns
import pdb
import torch.nn.functional as F
import json

from func_timeout import func_timeout, FunctionTimedOut



def get_wcd(x, true_wcd = True, model = None, grid_size = 6,K=4):
    
    if true_wcd:
        return compute_true_wcd(x,K=K)
    else:
        return model(x.float().cuda()).item()

def explore_changes(x, grid_size=6, init_wcd=0, pred_model=None, use_true_wcd=False, actions_allowed = "ALL_MODS",K=4):
    rows, cols = grid_size, grid_size
    min_wcd = init_wcd  # Initialize with positive infinity
    x_i = x.clone()
    result = decode_mdp_design(x_i.cpu(),return_grid = True, K=K)
    grid = result[1]
    encoding = result[0]
    protected_pos = encoding["goal_positions"]+[encoding["start_pos"]]
    gamma = K
    
    best_x_i = x
    

    def explore_helper(i, j, new_i, new_j):
        nonlocal min_wcd, best_x_i, x_i
        
        
        if check_design_is_valid(grid, is_grid = True, K =K)[0]:
            x_i = encode_from_grid_to_x(grid,gamma=K)[:,0:4,:,:]
            wcd = get_wcd(x_i, true_wcd=use_true_wcd, model=pred_model, grid_size=grid_size,K=K)
            # if wcd is None:
            #     pdb.set_trace()
            if wcd < min_wcd:
                min_wcd = wcd
                best_x_i = x_i.clone()
        else:
            # Revert the swap if the new design is not valid
            grid[i][j], grid[new_i][new_j] = grid[new_i][new_j], grid[i][j]
    
    for i in range(rows):
        for j in range(cols):
            if (i,j) in protected_pos:
                continue

            current_value = grid[i][j]
            
            if actions_allowed == "BLOCKING_ONLY" and current_value == 'X': # no unblocking is allowed
                continue 
            
            if actions_allowed == "UNBLOCKING_ONLY" and current_value != 'X': # blocking is allowed
                continue 
            
            grid[i][j] = ' ' if current_value == 'X' else 'X'

            # Explore the change
            explore_helper(i, j, i, j)

            # Revert the change
            grid[i][j] = current_value
                

    return best_x_i, min_wcd


def explore_changes_with_timeout(x, grid_size=6,init_wcd=0,  pred_model=None, use_true_wcd=False, timeout= 30, actions_allowed = "ALL_MODS",K=4):
    start_time = time.time()
    try:
        result = func_timeout(timeout, explore_changes, args=(x, grid_size,init_wcd, pred_model, use_true_wcd, actions_allowed,K))
        elapsed_time = time.time() - start_time
        _, cur_wcd = result
        return result, elapsed_time, cur_wcd
    except FunctionTimedOut:
        elapsed_time = time.time() - start_time
        print(f"Function timed out after {elapsed_time:.2f} seconds.")
        # Return the preliminary values on timeout
        return (x, init_wcd), elapsed_time, init_wcd  # Return init_wcd again to signify that it's the last known value

def iterative_greedy_search_with_timeout(initial_x,  budget=[1,2], grid_size=6, pred_model=None, 
                                         use_true_wcd=False, timeout = 30, actions_allowed = "BOTH", K=4):
    current_x = initial_x.clone()
    total_elapsed_time = 0
    
    budgets = []
    init_wcd = compute_true_wcd(initial_x.cpu(),K=K)
    cur_wcd = init_wcd
    env_i_s =[initial_x.cuda()]
    env_true_wcd = [init_wcd]
    
    budget_times= []
    budget_num_changes = []
    budget_wcd_change = []
    
    print("Budget",np.sum(budget))
    for iteration in range(np.sum(budget).astype(int)):
        print(f"Iteration {iteration + 1}: Current WCD ",cur_wcd )
        # Perform one iteration of greedy search with timeout
        try:
            (new_x, cur_wcd), elapsed_time, last_valid_wcd = explore_changes_with_timeout(current_x, 
                                                                                          grid_size, cur_wcd, 
                                                                                          pred_model, use_true_wcd, timeout, actions_allowed,K)
            total_elapsed_time += elapsed_time
            
        except FunctionTimedOut:
            total_elapsed_time += timeout
            print("Iteration timed out. Returning last valid WCD.")
            break

        
        budget_times.append(total_elapsed_time) 
        
        x_changes = new_x.cpu()[:, 1, :, :]-initial_x.cpu()[:, 1, :, :]
        blockings = (x_changes==1).sum(axis=(1, 2)).item()
        removals = (x_changes==-1).sum(axis=(1, 2)).item()
        
        budget_num_changes.append([blockings,removals])
        
        if actions_allowed != "BOTH":
            if blockings>=budget[0]:
                actions_allowed = "UNBLOCKING_ONLY"
            if removals>=budget[1]:
                actions_allowed = "BLOCKING_ONLY"
        
        
        final_wcd = compute_true_wcd(new_x.cpu(),K=K)
        budget_wcd_change.append(init_wcd - final_wcd)
        env_i_s.append(new_x.clone().cuda())
        env_true_wcd.append(final_wcd)
        
        # Check if there was any change in the iteration
        if torch.equal(current_x.cuda(), new_x.cuda()) or cur_wcd <= 0.1:
            print("No further changes. Stopping.", "wcd",cur_wcd )
            break
        # Use the result of the current iteration as input to the next
        current_x = new_x.clone()
    
    
    if iteration<np.sum(budget): # there was a timeout
        target_length = np.sum(budget).astype(int)
        # Filling up 'budget_times' and 'wcd_change' with their last value to match the length of 'budgets'
        budget_num_changes.extend([budget_num_changes[-1]] * (target_length - len(budget_num_changes)))
        budget_wcd_change.extend([budget_wcd_change[-1]] * (target_length - len(budget_wcd_change)))
        for _ in range((target_length - len(budget_times))):
            if total_elapsed_time>= timeout:
                budget_times.extend([timeout])
            else:
                budget_times.extend([budget_times[-1]])
        
    
    return budget_num_changes, budget_wcd_change, budget_times, env_i_s,env_true_wcd


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Simulate data")
    parser.add_argument(
        "--K",
        type=int,  # Ensure that the input is expected to be a float
        default=8,  # Set the default value to 0
        help="User model parameter",
    )
    
    parser.add_argument(
        "--grid_size",
        type=int,  # Ensure that the input is expected to be a int
        default=6,  # Set the default value to 1
        help="Maximum grid size.",
    )
    
    parser.add_argument(
        "--max_iter",
        type=int,  # Ensure that the input is expected to be a int
        default=20,  # Set the default value to 1
        help="Maximum number of iterations.",
    )
    parser.add_argument(
        "--timeout_seconds",
        type=int,  # Ensure that the input is expected to be a int
        default=600,  # Set the default value to 1
        help="Time to allow per experiment",
    )
    
    parser.add_argument(
        "--experiment_label",
        type=str,  # Ensure that the input is expected to be a string
        default="test",  # Set a default label for the experiment
        help="Label for the current experiment run. Default is 'default_experiment'.",
    )
    
    parser.add_argument(
        "--experiment_type",
        type=str,  # Ensure that the input is expected to be a str
        default="BOTH_UNIFORM_GREEDY_PRED_WCD",  # Set the default value to 1
        # choices = ["ALL_MODS"],
        choices = ["BLOCKING_ONLY_GREEDY_TRUE_WCD","BLOCKING_ONLY_GREEDY_PRED_WCD","ALL_MODS_GREEDY_TRUE_WCD", "ALL_MODS_GREEDY_PRED_WCD",
                  "BOTH_UNIFORM_GREEDY_TRUE_WCD", "BOTH_UNIFORM_GREEDY_PRED_WCD"],
        help="Either BLOCKING_ONLY or ALL_MODS for all modifications.",
    )
    
    parser.add_argument(
        "--ratio",
        type=str,  # Ensure that the input is expected to be a str
        default="1_3",  # Set the default value to 1:1
        # choices = ["ALL_MODS"],
        choices = ["1_1","1_2","1_3","3_1","1_5","5_1","2_1","3_2"],
        help="Either BLOCKING_ONLY or ALL_MODS for all modifications.",
    )

    parser.add_argument(
        "--start_index",
        type=int,  # Ensure that the input is expected to be a float
        default=0,  # Set the default value to 1
        help="Starting index for the number of instances",
    )
    
    args = parser.parse_args()
    K = args.K
    grid_size = args.grid_size
    experiment_label = args.experiment_label
    experiment_type = args.experiment_type
    _label = "_best"
    
    max_iter = args.max_iter
    
    dataset_label = f"data/grid{grid_size}/model_training/dataset_{grid_size}_K{K}{_label}.pkl"
    model_label = f"models/wcd_nn_model_{grid_size}_K{K}{_label}.pt" 

    with open(dataset_label, "rb") as f:
        loaded_dataset = pickle.load(f)
        
    experiment_label=experiment_type

    device ="cuda:0"
    model = torch.load(model_label)
    model = model.to(device).eval()

    true_wcds_per_cost=[]
    wcds_per_cost = []
    gammas = []
    times = []
    costs = []
    realized_budgets = []
    max_budgets = []
    
    eval_changes = [1,3,5,7,9,11,13,15,17,19]
    
    times = []
    all_wcd_changes = []
    budgets =[]
    gammas = []
    true_wcds_list = []
    
    if "BLOCKING_ONLY" in experiment_label:
        blocking_rat = 1
        unblocking_rat = 0
        actions_allowed = "BLOCKING_ONLY"
        eval_changes = [19]
    elif "ALL_MODS" in experiment_label: #ALL Modifications are allowed
        blocking_rat = int(args.ratio[0])
        unblocking_rat = int(args.ratio[2])
        actions_allowed = "ALL_MODS"
        
    else:
        blocking_rat = 1.01# hack to handle the scenario wher budget = 1
        unblocking_rat = 1
        actions_allowed = "BOTH" # ratio does not matter here -- the total budget is what matters
        eval_changes = [19]
    
    
    
    
    data_storage_path = f"baselines/data/grid{grid_size}/K{K}/timeout_{args.timeout_seconds}"
    create_folder(data_storage_path)
    data_storage_path = f"{data_storage_path}/{experiment_label}"
    create_folder(data_storage_path)
    create_folder(data_storage_path+"/individual_envs/")
    create_folder(data_storage_path+"/final_envs/")
    
    if  "ALL_MODS" in experiment_type: #ALL Modifications are allowed
        ratio = f"ratio_{blocking_rat}_{unblocking_rat}"
        data_storage_path = f"{data_storage_path}/{ratio}/"
        create_folder(data_storage_path)
    
    max_index = len(loaded_dataset)
    interval = 1
    max_index = np.min([args.start_index+(interval*2000),len(loaded_dataset)])
    
    print("Interval : ",interval)
    
    for i in range(args.start_index, len(loaded_dataset),interval):
        if os.path.exists(f"{data_storage_path}/individual_envs/env_{i}.json"):# skip if exists
            continue #this is already ran
            
        env_wcd_changes = []
        env_budgets = []
        env_times = []
        env_max_budget =[]

        x, y = loaded_dataset[i]  # Get a specific data sample

        x = x.unsqueeze(0).float().cuda()

        num_changes =[]
        
        print("Environment : ",i)

        for max_budget in eval_changes:
                print("Original X:, WCD = ",model(x).item())
                # plot_grid(decode_grid_design(x.cpu().squeeze(),return_map= True).tolist())
                best_wcd = []
                invalid_envs_collection =[]
                true_wcds = []

                max_changes_dist=np.round([(blocking_rat*max_budget)/(unblocking_rat+blocking_rat),
                                           (unblocking_rat*max_budget)/(unblocking_rat+blocking_rat) ]).tolist()

                max_iter = max_budget

                budget_num_changes, budget_wcd_change, budget_times, x_envs,true_wcds = iterative_greedy_search_with_timeout(
                    x,  budget=max_changes_dist, grid_size=grid_size, pred_model=model, K=K,
                    use_true_wcd=True if "TRUE_WCD" in experiment_type else False , 
                    timeout = args.timeout_seconds, 
                    actions_allowed = actions_allowed
                )

                env_times=budget_times
                env_wcd_changes=budget_wcd_change


                if len(x_envs) ==1:
                        continue

                num_changes = budget_num_changes
                env_max_budget= max_changes_dist
                env_budgets=budget_num_changes

                if "TRUE_WCD" in experiment_type: #store for training future models
                    update_or_create_dataset(f"data/grid{grid_size}/model_training/simulated_greedy_valids_final{grid_size}_K{K}_{experiment_label}.pkl"
                                             , 
                                             x_envs, true_wcds)


        env_dict = {
            "env_id":int(i),
            "times":env_times,
            "wcd_changes":env_wcd_changes,
            "budgets":env_budgets,
            "max_budgets":eval_changes,
            "num_changes":num_changes}

        # print(env_dict)
        with open(f"{data_storage_path}/individual_envs/env_{i}.json", "w") as json_file:
            json.dump(env_dict, json_file, indent=4)

        torch.save( x_envs[-1],f"{data_storage_path}/final_envs/env_{i}_budget{max_budget}.pt")



     