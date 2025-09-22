"""
Optimization Script for Overcooked-AI Goal Recognition Design

Author: Robert Kasumba (rkasumba@wustl.edu)

This script implements the core optimization process for minimizing Worst-Case Distance (WCD)
in Overcooked-AI environments using gradient descent with the trained CNN oracle.

WHY THIS IS NEEDED:
- Goal recognition design requires finding environment modifications that minimize WCD
- Direct optimization is intractable due to discrete environment spaces
- Gradient-based optimization enables efficient search through continuous relaxations
- Lagrangian multipliers handle constraints on the number of allowed modifications

HOW IT WORKS:
1. Loads the trained CNN oracle and environment dataset
2. For each environment, applies gradient descent to minimize WCD + cost penalties
3. Uses Lagrangian multipliers to enforce constraints on modifications:
   - CONSTRAINED: Separate constraints for OT (Onion+Tomato) and SPD (Soup+Plate+Dish) changes
   - UNCONSTRAINED: Single constraint on total number of changes
4. Tracks optimization progress, WCD changes, and computation time
5. Saves results for each environment and lambda combination

OPTIMIZATION PROCESS:
- Applies gradient descent to minimize: WCD + λ₁×OT_changes + λ₂×SPD_changes
- Uses Manhattan distance to measure modification costs
- Supports both optimal (γ=0.99999) and suboptimal (γ∈[0.65,1.0]) agent behavior
- Iterates through different lambda values to explore trade-offs

USAGE:
    python run_optmize_wcd.py --cost 0 --start_index 0 --max_grid_size 6 --experiment_label test --optimality OPTIMAL --experiment_type CONSTRAINED

OUTPUT:
    JSON files with optimization results for each environment in data/grid{size}/optim_runs/{experiment_type}/langrange_values/
"""

import sys
# sys.path.insert(0, "../")
# sys.path.insert(0, "../../")
import torch
from torch.utils.data import Dataset
from utils import *
import random
import matplotlib.pyplot as plt
from wcd_simulation_utils import *
from torchvision.models import resnet50, resnet18
import argparse
import traceback
import time
import json

def manhattan_distance(x1, x2):
    """
    Compute the Manhattan distance between the positions of 1s in two tensors.

    Args:
    - x1 (torch.Tensor): Input tensor 1.
    - x2 (torch.Tensor): Input tensor 2.

    Returns:
    - torch.Tensor: Manhattan distance between the positions of 1s in the input tensors.
    """

    # Create a grid of indices
    indices = torch.arange(x1.numel()).reshape(x1.shape).cuda()

    # Multiply indices by the tensors (without modifying inplace)
    x1_indices = x1 * indices
    x2_indices = x2 * indices

    # Compute the absolute sum of the difference
    manhattan_distance = torch.abs(torch.sum(x1_indices - x2_indices))

    return manhattan_distance

def compute_loss(x, model, x_0, lamda=[0.0]):
    """
    Compute the optimization loss combining WCD prediction and modification costs.
    
    Args:
        x: Current environment tensor (modified)
        model: Trained CNN oracle for WCD prediction
        x_0: Original environment tensor (unmodified)
        lamda: Lagrangian multipliers for constraint penalties
    
    Returns:
        loss: Total loss (WCD + constraint penalties)
        wcd: Predicted WCD value
    """
    
    # Forward pass through CNN to get WCD prediction
    output = model(x)
    wcd = output.mean()  # Average WCD across batch
    
    if len(lamda) == 1:
        # Unconstrained optimization: single penalty for total changes
        # L1 norm of differences in environment channels (0-6)
        sim_loss = torch.norm((x[:, :7, :, :] - x_0[:, :7, :, :]), p=1) + 5e-3
        loss = (wcd + lamda[0] * sim_loss) 
    else:
        # Constrained optimization: separate penalties for OT and SPD changes
        
        # OT (Onion + Tomato) constraint: channels 3 and 4
        sim_loss_OT = (manhattan_distance(x[:, 3, :, :], x_0[:, 3, :, :]) + 
                      manhattan_distance(x[:, 4, :, :], x_0[:, 4, :, :]))
        
        # SPD (Soup + Plate + Dish) constraint: channels 1, 5, and 2
        sim_loss_SPD = (manhattan_distance(x[:, 1, :, :], x_0[:, 1, :, :]) + 
                       manhattan_distance(x[:, 5, :, :], x_0[:, 5, :, :]) + 
                       manhattan_distance(x[:, 2, :, :], x_0[:, 2, :, :]))
        
        # Combined loss: WCD + OT penalty + SPD penalty
        loss = (wcd + lamda[0] * sim_loss_OT + lamda[1] * sim_loss_SPD) 
    
    return loss, wcd
    

def compute_gradients (x,model, x_0, lamda=0.0):
    x.requires_grad = True

    loss,wcd = compute_loss(x,model, x_0, lamda=lamda)
    loss.backward()
    
    # Access gradients for x
    gradients = x.grad
    x.requires_grad = False
    return loss.item(),wcd.item(),gradients


def randomly_choose_x_i(possible_x_is,k=1,grid_size=6):
    """
    This function shuffles the possible_envs list, checks each environment,
    and returns True if a valid environment is found. Otherwise, it returns False.
    """
    if k!=1:
        random.shuffle(possible_x_is)  # Shuffle the list to pick envs randomly
    
    for x_i in possible_x_is:
        env = decode_env(x_i.squeeze(),grid_size=grid_size)
        if check_env_is_valid(env):  # If a valid env is found, return True
            return True, x_i
        else:
            # print_grid(env)
            print("Invalid found")
    
    # If loop completes, no valid env was found, so return False
    return False,x_i

def max_grad_update_single_channel(x_i, x_grad, random_channel = False, black_list_pos =[],model=None, x_init = None, lamda=None, best_loss = None, grid_size=6):
    updated_x_i = x_i.clone()
    
    # Find the channel with the maximum gradient

    x_grad_ = x_grad.view((x_grad.size(0), x_grad.size(1),-1))[:, 0:7, :] # channels 7,8,9,10,11,12 cannot change. i.e agent pos and the goals
    x_grad_abs = x_grad[:, 0:7, :].abs()
    
    # the channel for X's (blocked) and space " " channels should not be used in computing the max
    x_grad_abs[:,0,:] =  -float('inf') # grad in blocked (X) channel is set tp -ve infinity
    x_grad_abs[:,6,:] =  -float('inf') # grad in space (" ") channel is set tp -ve infinity
    
    x_grad_abs = x_grad_abs.view(1, -1)
    updated_pos = -1 # default if no change is made
    for pos in black_list_pos:# disqualify these
        x_grad_abs[0,pos] = -float('inf')
        
    possible_x_is =[]
    
    invalid_x_is =[]
    
    found_x_i = False
   
    for i in range(x_grad_abs.shape[1]):
        
        max_position = torch.argmax(x_grad_abs)
        # Convert the flattened index to 4D coordinates
        channel_dim = max_position // (x_grad.shape[2] * x_grad.shape[3])
        max_position_remainder = max_position % (x_grad.shape[2] * x_grad.shape[3])
        height_dim = max_position_remainder // x_grad.shape[3]
        width_dim = max_position_remainder % x_grad.shape[3]
    
        max_position_4d = (
            0,  # Batch dimension
            channel_dim,  # Channel dimension
            height_dim,  # Height dimension
            width_dim  # Width dimension
        )
        
        if torch.sign(x_grad[max_position_4d]).float() != 0:
            updated_x_i[max_position_4d] = torch.sign(x_grad[max_position_4d]).float()
          # Clip the values to ensure they remain within the range of 0 or 1
        channel = max_position_4d[1]
        updated_x_i[:, channel, :, :] = torch.clamp(updated_x_i[:, channel, :, :], 0, 1)
        
        x_grad_abs[0,max_position] = -float('inf')
        
        # Check if there is more than one 1 in the channel
        num_ones = updated_x_i[:, channel, :, :].sum()
  
        if (num_ones > 1).any() and (channel in [1,2,3,4,5]):
            # Remove the original one and set it to 0
            updated_x_i[:, channel, :, :] = torch.zeros_like(updated_x_i[:, channel, :, :])
            updated_x_i[max_position_4d] = 1
        elif (num_ones == 0).any():
            x_grad_pos = torch.clamp(x_grad[:, channel, :, :], min=0).unsqueeze(0)
            max_values, max_indices = x_grad_pos[:, 0, :, :].abs().view(x_grad.size(0), -1).max(dim=1)
            # Convert max_indices to 2D coordinates
            max_indices_2d = torch.stack((max_indices // x_grad.size(2), max_indices % x_grad.size(2)), dim=1).flatten()
            updated_x_i[:, channel, max_indices_2d[0], max_indices_2d[1]] = 1
            
        if updated_x_i[max_position_4d] != x_i[max_position_4d]:
            updated_pos = max_position
            curr_loss,_ = compute_loss(updated_x_i,model, x_init, lamda=lamda)
            if curr_loss <= best_loss:
                posits = []
                collision = False
                for ch in [1,2,3,4,5]:
                    try:
                        coordinates = torch.where(updated_x_i[0, ch, :, :] == 1)
                        pos = (coordinates[0].item(), coordinates[1].item())
                        if pos in posits:
                            posits.append(pos)
                            collision= True
                            print("Collision detected")
                    except:
                        collision= True
                # print(model(updated_x_i).item(), best_wcd,collision)
                
                if not collision:
                    # possible_x_is.append(updated_x_i)
                    env = decode_env(updated_x_i.squeeze(),grid_size=grid_size)
                    if check_env_is_valid(env):  # If a valid env is found, return True
                        found_x_i = True
                        break


        updated_x_i= x_i.clone()
        
                
    if found_x_i:
        return updated_x_i, channel, updated_pos,invalid_x_is
    else:
        return updated_x_i, -1, -1,invalid_x_is

def minimize_wcd(model, x, cost=0.0,grid_size=6, max_iter = 10, verbose= False):

    x_i = x.clone()
    lowest_loss,_= compute_loss(x, model, x_0=x,  lamda = cost )
    best_x_i = x_i.clone()
    no_progress = 0
    convergence_threshold = 1e-5
    prev_loss = float('inf')
    wcd =1000
    invalid_envs = [] # store environment designs that successfully fooled the model

    channel_tracker = 0
    prev_channel = -1
    black_list_pos =[]
    wcds = [lowest_loss]
    x_envs = [x_i]
    true_wcds =[compute_true_wcd(x_i,grid_size=grid_size)]
    iters = [0]
    print("Pred:",wcds[-1]," True",true_wcds[-1])
    cumulative_time = 0
    for i in range(1,max_iter+1):
        # Start the timer
        start_time = time.time()
        
        loss, wcd, x_grad = compute_gradients(x_i, model, x_0=x,  lamda = cost )
        
        x_i,channel,updated_pos,invalid_x_is = max_grad_update_single_channel(x_i,-1*x_grad,
                                                                              random_channel=False,black_list_pos=black_list_pos,model=model,
                                                                              best_loss=lowest_loss, x_init = x, lamda = cost ,grid_size=grid_size)
        if updated_pos == -1: # there was no update
            break
            
        black_list_pos.append(updated_pos)
        wcd = model(x_i).item()
        invalid_envs+=invalid_x_is

        if wcd <lowest_loss:
            env = decode_env(x_i.squeeze(),grid_size=grid_size)
            if check_env_is_valid(env):
                best_x_i = x_i.clone()
                lowest_loss = wcd
                no_progress =0
            else: #invalid environment returned
                # print("We got invalid at i=",i)
                # print(decode_env(x_i.squeeze(),grid_size=grid_size))
                invalid_envs.append(x_i)
                no_progress += 1
            x_i = best_x_i
        else:
            no_progress += 1

        if no_progress >10:
            break
            
        # End the timer
        end_time = time.time()
        
        # Calculate the time taken in seconds
        time_taken_ms = (end_time - start_time)
        cumulative_time += time_taken_ms
            
       
        
        if i%15==1 and verbose:
            # print(decode_env(x_i.squeeze(),grid_size=grid_size))
            print((x==x_i).all(), model(x_i).item(), model(x).item())
            
            true_wcds.append(compute_true_wcd(x_i,grid_size=grid_size))
            wcds.append(wcd)
            iters.append(i)
            x_envs.append(x_i)
            
            print("i = ",i,"Pred:",wcd," True",true_wcds[-1])
    
    # record final WCDs
    true_wcds.append(compute_true_wcd(x_i,grid_size=grid_size))
    wcds.append(wcd)
    iters.append(i)
    x_envs.append(x_i)
   
    return best_x_i,invalid_envs,wcds,true_wcds,x_envs, iters, cumulative_time


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
        "--start_index",
        type=int,  # Ensure that the input is expected to be a float
        default=0,  # Set the default value to 1
        help="Starting index for the number of instances",
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
        "--optimality",
        type=str,  # Ensure that the input is expected to be a string
        default="OPTIMAL",  # Set a default label for the experiment
        choices =["SUBOPTIMAL","SUBOPTIMAL"],
        # choices =["ALL_MODS_EXHUASTIVE","ALL_MODS_GREEDY_TRUE_WCD","ALL_MODS_GREEDY_PRED_WCD"],
        help="Behavior optimality'.",
    )
    
    parser.add_argument(
        "--experiment_type",
        type=str,  # Ensure that the input is expected to be a string
        default="CONSTRAINED",  # Set a default label for the experiment
        choices =["CONSTRAINED","UNCONSTRAINED"],
        # choices =["ALL_MODS_EXHUASTIVE","ALL_MODS_GREEDY_TRUE_WCD","ALL_MODS_GREEDY_PRED_WCD"],
        help="Behavior optimality'.",
    )

    
    
    args = parser.parse_args()
    
    cost = args.cost

    grid_size = args.max_grid_size
    base_data_path = f"./data/grid{grid_size}" # for reading files
    experiment_label = args.experiment_label
    experiment_type = args.experiment_type
    
    optimality = args.optimality
    optim_folder = "optim_runs" if optimality == "OPTIMAL" else "suboptimal_runs" 
    with open(f"{base_data_path}/model_training/dataset_{grid_size}_{args.experiment_label}.pkl", "rb") as f:
        loaded_dataset = pickle.load(f)
    
    device ="cuda:0"
    model = torch.load(f"models/wcd_nn_oracle_random_{grid_size}_{args.experiment_label}.pt")
    model = model.to(device).eval()
    
    
    
    true_wcds_per_cost=[]
    wcds_per_cost = []
    gammas = []
    interval = 2
    times_taken = []
    max_budgets = []
    wcd_changes = []
    budget_changes =[]
    pred_wcds = []
    gammas = []
    
    costs = [0,0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1.0,2,5] # lagrangian multipliers
    
    if experiment_type == "CONSTRAINED":
        costs = [0,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1.0,5]
        costs = [0,0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1.0,2,5] # with twos
        lang_multipliers = [[i,j] for i in costs for j in costs]
    else:
        lang_multipliers = [[i] for i in costs]
    

    # create basic folder
    create_folder(f"data/grid{grid_size}/{optim_folder}/{experiment_type}")
    create_folder(f"data/grid{grid_size}/{optim_folder}/{experiment_type}/langrange_values")
    
    create_or_update_list_file(f"data/grid{grid_size}/{optim_folder}/{experiment_type}/budget_{grid_size}_{experiment_label}.csv",[lang_multipliers])
    
    
    max_index = len(loaded_dataset)
    
    max_index = np.min([args.start_index+(20*200),len(loaded_dataset)])
    
    # Main optimization loop: process environments in batches of 20
    for i in range(args.start_index, max_index, 20):
        try:
            print("Environment; ", i)
            x, y = loaded_dataset[i]  # Get a specific data sample
            
            # Initialize result dictionary for this environment
            env_dict = {
                "env_id": i,
                "lambdas": []
            }
                
            # Set gamma value based on agent optimality
            if optimality == "OPTIMAL":
                gamma = 0.99999  # Near-optimal agent behavior
            else:
                gamma = randomly_choose_gamma(ranges=[(0.65, 1.0)], probabilities=[1.0])  # Suboptimal behavior
            
            # Set gamma value across all positions in the environment
            x[8, :, :] = torch.full((grid_size, grid_size), gamma).float()
            x = x.unsqueeze(0).float().cuda()  # Add batch dimension and move to GPU

            print("Original X:, WCD = ", model(x).item())
            
            # Initialize tracking variables for this environment
            best_wcd = []
            invalid_envs_collection = []
            true_wcds = []
            budget_time_taken = []
            budget = []
            wcd_change = []
            max_budget = 14  # Maximum optimization iterations
            num_changes = []
            
            # Try different lambda (cost) values for this environment
            for cost in lang_multipliers:
                print("COST === ", cost)
                # Run optimization with current lambda values
                best_x_i, invalid_envs, wcds, true_wcds, x_envs, iters, time_taken = minimize_wcd(
                    model, x, cost=cost, grid_size=grid_size, max_iter=max_budget
                )

                # Skip if optimization didn't produce valid results
                if len(x_envs) == 1:
                    continue
                
                # Record optimization results
                budget_time_taken.append(time_taken)
                wcd_change.append(true_wcds[0] - true_wcds[-1])  # WCD improvement
                num_changes.append(np.sum(decode_env(x.squeeze()) != decode_env(best_x_i.squeeze())))
                
                # Store results in dictionary
                env_dict["lambdas"].append({
                    "lambdas": cost,
                    "wcd_change": wcd_change[-1],
                    "num_changes": compute_changes(decode_env(x.squeeze()), decode_env(best_x_i.squeeze())),
                    "time_taken": time_taken
                })
                
                # Save intermediate results to JSON file
                with open(f"{base_data_path}/{optim_folder}/{experiment_type}/langrange_values/env_{i}.json", "w") as json_file:
                    json.dump(env_dict, json_file, indent=4)
                
            # Store results for this environment
            times_taken.append(budget_time_taken)
            wcd_changes.append(wcd_change)
            budget_changes.append(num_changes)
            gammas.append(gamma)
            
        
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()

    # plot_combined(wcds_per_cost,true_wcds_per_cost,gammas)
    