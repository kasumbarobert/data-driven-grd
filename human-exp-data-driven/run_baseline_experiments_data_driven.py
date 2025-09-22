import sys
# sys.path.insert(0, "./")
# sys.path.insert(0, "../../")
import torch
from torch.utils.data import Dataset
from utils_human_exp import *
import random
import matplotlib.pyplot as plt

from torchvision.models import resnet50, resnet18
import argparse
import traceback
import time
import seaborn as sns
import pdb
import torch.nn.functional as F
import traceback
import json

from func_timeout import func_timeout, FunctionTimedOut

human_model = torch.load('models/human_model_grid6.pt', map_location=torch.device('cpu'))
human_model.eval()

def get_wcd(x, true_wcd = True, model = None, grid_size = 6,assumed_behavior="OPTUMAL",search_depth=10):
    
    if true_wcd:
        if assumed_behavior:
            return compute_humal_model_wcd(x.squeeze(0),model=human_model,search_depth=search_depth)
        else:
            return compute_true_wcd(x.squeeze(0))
    else:
        return model(x.float().cuda()).item()

def explore_changes(x, grid_size=6, init_wcd=0, pred_model=None, use_true_wcd=False, actions_allowed = "ALL_MODS",assumed_behavior="OPTUMAL",search_depth=10):
    rows, cols = grid_size, grid_size
    min_wcd = get_wcd(x.cpu(), true_wcd=use_true_wcd, model=pred_model, grid_size=grid_size,assumed_behavior =assumed_behavior,search_depth=search_depth)  # Initialize with positive infinity
    
    if min_wcd is None:
        return x,None
    
    x_i = x.clone()
    grid = decode_grid_design(x_i.squeeze(0).cpu(),return_map = True)

    n, goal_pos, blocked_pos ,start_position ,spaces_pos = decode_grid_design(x_i.squeeze(0).cpu(),return_map = False)
    print("Goal Positions",goal_pos)
    protected_pos = goal_pos+[start_position]

    
    best_x_i = x
    

    def explore_helper(i, j, new_i, new_j):
        nonlocal min_wcd, best_x_i, x_i
        
        
        if check_design_is_valid(grid, is_grid = True)[0]:
            x_i = encode_from_grid_to_x(grid)[:,0:4,:,:]
            wcd = get_wcd(x_i, true_wcd=use_true_wcd, model=pred_model, grid_size=grid_size,assumed_behavior =assumed_behavior,search_depth=search_depth)
            if wcd is not None and wcd>=0:
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


def explore_changes_with_timeout(x, grid_size=6,init_wcd=0,  pred_model=None, use_true_wcd=False, timeout= 30, actions_allowed = "ALL_MODS",assumed_behavior="OPTIMAL",search_depth=10):
    start_time = time.time()
    try:
        result = func_timeout(timeout, explore_changes, args=(x, grid_size,init_wcd, pred_model, use_true_wcd, actions_allowed,assumed_behavior,search_depth))
        elapsed_time = time.time() - start_time
        _, cur_wcd = result
        return result, elapsed_time, cur_wcd
    except FunctionTimedOut:
        elapsed_time = time.time() - start_time
        print(f"Function timed out after {elapsed_time:.2f} seconds.")
        # Return the preliminary values on timeout
        return (x, init_wcd), elapsed_time, init_wcd  # Return init_wcd again to signify that it's the last known value

def iterative_greedy_search_with_timeout(initial_x,  budget=[1,2], grid_size=6, pred_model=None, use_true_wcd=False, timeout = 30, actions_allowed = "BOTH", assumed_behavior="OPTIMAL",search_depth = 10):
    current_x = initial_x.clone()
    total_elapsed_time = 0
    
    budgets = []
    init_wcd = compute_humal_model_wcd(initial_x.squeeze().cpu(),model=human_model,search_depth=search_depth)
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
            (new_x, cur_wcd), elapsed_time, last_valid_wcd = explore_changes_with_timeout(current_x, grid_size, cur_wcd, 
                                                                                          pred_model, use_true_wcd, timeout, actions_allowed,assumed_behavior,search_depth)
            total_elapsed_time += elapsed_time
            
        except FunctionTimedOut:
            total_elapsed_time += timeout
            print("Iteration timed out. Returning last valid WCD.")
            break
            
        
        if cur_wcd is None:
            raise ValueError("The environment has invalid Human WCD")

        
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
        
        
        final_wcd = compute_humal_model_wcd(new_x.squeeze().cpu(),model=human_model,search_depth=search_depth) #evaluation is based on human model
        if final_wcd is None or final_wcd <0:
            print(decode_grid_design(new_x.squeeze().cpu(),return_map = True))
            break
        
        
        budget_wcd_change.append(init_wcd - final_wcd)
        env_i_s.append(new_x.clone().cuda())
        env_true_wcd.append(final_wcd)
        print(budget_wcd_change)
        
        # Check if there was any change in the iteration
        if torch.equal(current_x.cuda(), new_x.cuda()) or cur_wcd <= 0.1:
            print("No further changes. Stopping.", "wcd",cur_wcd )
            break
        # Use the result of the current iteration as input to the next
        current_x = new_x.clone()
    
    
    if len(budget_wcd_change) == 0:
        budget_wcd_change.append(0)
        budget_num_changes.append([0,0])
        budget_times.append(timeout*2)
    
    if iteration<np.sum(budget): # there was a timeout
        target_length = np.sum(budget).astype(int)
        # Filling up 'budget_times' and 'wcd_change' with their last value to match the length of 'budgets'
        budget_num_changes.extend([budget_num_changes[-1]] * (target_length - len(budget_num_changes)))
        budget_wcd_change.extend([budget_wcd_change[-1]] * (target_length - len(budget_wcd_change)))
        
        if torch.equal(current_x.cuda(), new_x.cuda()) or cur_wcd <= 0.1:
            for _ in range((target_length - len(budget_times))):
                budget_times.extend([budget_times[-1]])
        else:
            for _ in range((target_length - len(budget_times))):
                budget_times.extend([timeout])
        
    
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
        "--grid_size",
        type=int,  # Ensure that the input is expected to be a int
        default=6,  # Set the default value to 1
        choices =[6,10],
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
        default=1800,  # Set the default value to 1
        help="Time to allow per experiment",
    )
    
    parser.add_argument(
        "--experiment_label",
        type=str,  # Ensure that the input is expected to be a string
        default="test",  # Set a default label for the experiment
        help="Label for the current experiment run. Default is 'default_experiment'.",
    )
    
    parser.add_argument(
        "--assumed_behavior",
        type=str,  # Ensure that the input is expected to be a str
        default="HUMAN",  # Set the default value to 1
        # choices = ["ALL_MODS"],
        choices = ["OPTIMAL","HUMAN"]
    )
    
    parser.add_argument(
        "--experiment_type",
        type=str,  # Ensure that the input is expected to be a str
        default="BOTH_UNIFORM_GREEDY_TRUE_WCD",  # Set the default value to 1
        # choices = ["ALL_MODS"],
        choices = ["BLOCKING_ONLY_GREEDY_TRUE_WCD","BLOCKING_ONLY_GREEDY_PRED_WCD","ALL_MODS_GREEDY_TRUE_WCD", "ALL_MODS_GREEDY_PRED_WCD",
                  "BOTH_UNIFORM_GREEDY_TRUE_WCD", "BOTH_UNIFORM_GREEDY_PRED_WCD"],
        help="Either BLOCKING_ONLY or ALL_MODS for all modifications.",
    )

    
    args = parser.parse_args()
    grid_size = args.grid_size
    experiment_label = args.experiment_label
    experiment_type = args.experiment_type
    _label = "_best"
    
    human_model = torch.load(f'models/human_model_grid{grid_size}.pt', map_location=torch.device('cpu'))
    
    max_iter = args.max_iter
    
    dataset_label = f"data/grid{grid_size}/model_training/dataset_{grid_size}_may10.pkl"
    
    if args.assumed_behavior=="OPTIMAL":
        model_label = f"models/wcd_nn_model_{grid_size}{_label}_optimal.pt"
    else:
        model_label = f"models/wcd_nn_model_{grid_size}_human_may10.pt"

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
    
    search_depth = 10 if grid_size ==6 else 19
    
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
        blocking_rat = 3
        unblocking_rat = 5
        actions_allowed = "ALL_MODS"
        
    else:
        blocking_rat = 1.01# hack to handle the scenario wher budget = 1
        unblocking_rat = 1
        actions_allowed = "BOTH" # ratio does not matter here -- the total budget is what matters
        eval_changes = [19]
        
    
    data_storage_path = f"baselines/data/grid{grid_size}/timeout_{args.timeout_seconds}/{args.assumed_behavior}/{experiment_label}"
    create_folder(data_storage_path)
    create_folder(data_storage_path+"/individual_envs")
    create_folder(data_storage_path+"/final_envs")
    num_instances = 400
    env_ids_list = range(0, len(loaded_dataset),len(loaded_dataset)//num_instances)
    # env_ids_list = np.array([101, 108,  98,  13, 112,  29,  95,  77, 120,  88,  52, 126,  81,
    #    137, 100, 115, 136,  72,  14,   6, 130, 104,  76,  78,  83,  62,
    #     99, 116, 119,  38])*(len(loaded_dataset)//num_instances)
    for i in env_ids_list:
            try:
                env_wcd_changes = []
                env_budgets = []
                env_times = []
                env_max_budget =[]

                x, y = loaded_dataset[i]  # Get a specific data sample

                x = x.unsqueeze(0).float().cuda()

                num_changes =[]

                for max_budget in eval_changes:
                        print("Env",i, "Original X:, WCD = ",model(x).item())
                        # plot_grid(decode_grid_design(x.cpu().squeeze(),return_map= True).tolist())
                        best_wcd = []
                        invalid_envs_collection =[]
                        true_wcds = []

                        max_changes_dist=np.round([(blocking_rat*max_budget)/(unblocking_rat+blocking_rat),
                                                   (unblocking_rat*max_budget)/(unblocking_rat+blocking_rat) ]).tolist()

                        max_iter = max_budget

                        budget_num_changes, budget_wcd_change, budget_times, x_envs,true_wcds = iterative_greedy_search_with_timeout(
                            x,  budget=max_changes_dist, grid_size=grid_size, pred_model=model, 
                            use_true_wcd=True if "TRUE_WCD" in experiment_type else False ,
                            timeout = args.timeout_seconds, 
                            assumed_behavior=args.assumed_behavior,
                            actions_allowed = actions_allowed,search_depth=search_depth)

                        env_times= budget_times
                        env_wcd_changes=budget_wcd_change


                        if len(x_envs) ==1:
                                continue


                        num_changes =budget_num_changes
                        env_max_budget= max_changes_dist
                        env_budgets = budget_num_changes

                        if "TRUE_WCD" in experiment_type: #store for training future models
                                    update_or_create_dataset(f"data/grid{grid_size}/model_training/simulated_greedy_valids_final{grid_size}_{experiment_label}.pkl", 
                                                     x_envs, true_wcds)
                
                env_dict = {
                    "env_id":int(i),
                    "times":env_times,
                    "wcd_changes":env_wcd_changes,
                    "budgets":env_budgets,
                    "max_budgets":list(range(1,max_budget+1)),
                    "num_changes":num_changes
                }
                # print(env_dict)
                with open(f"{data_storage_path}/individual_envs/env_{i}.json", "w") as json_file:
                    json.dump(env_dict, json_file, indent=4)
                    
                torch.save( x_envs[-1],f"{data_storage_path}/final_envs/env_{i}_budget{max_budget}.pt") # store the final environments
                
                
            except Exception as e:
                print("Exception details:")
                traceback.print_exc()

            # break;



     