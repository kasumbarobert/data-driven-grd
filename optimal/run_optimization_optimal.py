import sys
# sys.path.insert(0, "./")
# sys.path.insert(0, "../../")
import torch
from torch.utils.data import Dataset
from utils import *
import random
import matplotlib.pyplot as plt

from torchvision.models import resnet50, resnet18
import argparse
import traceback
import time
import seaborn as sns
import pdb
import torch.nn.functional as F
import pynvml
import os
import gc
import json
import logging
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:526"
COMPUTE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("COMPUTE_DEVICE", COMPUTE_DEVICE, torch.cuda.is_available())

def create_large_tensor():
    size_gb = 6
    # Calculate the number of elements based on the desired size in gigabytes
    num_elements = int(size_gb * (1024 ** 3) / 4)  # Assuming each element is 4 bytes (float32)

    # Create a tensor with the specified number of elements
    large_tensor = torch.randn(num_elements).to(COMPUTE_DEVICE)
    
    return large_tensor
# large_tensor = create_large_tensor()
large_tensor = None
    
def compute_loss(x,model, x_0, lambdas =[0,0],max_changes_dist = [3, 5]):
    
    # Forward pass
    output = model(x.to(COMPUTE_DEVICE))

    # Define a scalar loss function
    lambda_1,lambda_2 = lambdas[0],lambdas[1]
    wcd = output.mean()
    
    x_changes = x[:, 1, :, :]-x_0[:, 1, :, :]
    

    blockings = torch.sum(F.softplus(x_changes))
    removals = torch.sum(F.softplus(torch.abs(-x_changes)))

    
    # print(max_changes_dist)
    
    if max_changes_dist == [-1,-1]: # no limits
        penalty_term = lambda_1*(blockings+removals)

        sim_loss = penalty_term
    else:
        penalty_term_1 = lambda_1 * blockings
        penalty_term_2 = lambda_2 * removals

        sim_loss = penalty_term_1 + penalty_term_2

    loss =(wcd +sim_loss) 
    
    del output, x_changes, blockings, removals #clear memory

    return loss,wcd.cpu()
  
def compute_gradients(x, model, x_0, lambdas=[0, 0], max_changes_dist=[3, 5], sensitivity_analysis = False, noise_level = 0.0):
    x.requires_grad = True
    
    # Zero gradients before backward pass
    model.zero_grad()

    loss, wcd = compute_loss(x, model, x_0, lambdas=lambdas, max_changes_dist=max_changes_dist)
    
    loss = loss.mean()  # Add this line to ensure that loss is a scalar
    
    # After the backward pass
    loss.backward()


    # Access gradients for x
    if sensitivity_analysis:
        gradients = x.grad + noise_level * torch.randn_like(x.grad)
    else:
        gradients = x.grad

    # Clear gradient computation graph
    model.zero_grad()

    x.requires_grad = False
    
    
    loss_val = loss.item()
    
    del loss
    
    return loss_val, wcd.item(), gradients.cpu()


def max_grad_update_single_channel(x_i, x_grad, black_list_pos =[],model=None, best_loss = None,grid_size=19,shortest_path_lens=None, init_x = None, lambdas = [0,0], max_changes_dist=[3,5] ):
    updated_x_i = x_i.clone()
    

    x_grad_ = x_grad.view((x_grad.size(0), x_grad.size(1),-1))[:, 0:2, :] # channels  2,3 cannot change
    x_grad_abs = x_grad[:, 0:2, :].cpu().abs().view(1, -1)
    updated_pos = -1 # default if no change is made
    
    for pos in black_list_pos:# disqualify these
        x_grad_abs[0,pos] = -float('inf')
        
    possible_x_is =[]
    
    invalid_x_is =[]
    
    found_x_i = False
    for i in range(x_grad_abs.shape[1]):
        max_position = torch.argmax(x_grad_abs).cpu()

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
        channel = max_position_4d[1]
        opposite_channel_4d = (
            0,  # Batch dimension
            1-channel_dim,  # Channel dimension in the opposite channel
            height_dim,  # Height dimension
            width_dim  # Width dimension
        )
        

        if torch.sign(x_grad[max_position_4d]).float() != 0:
            updated_x_i[max_position_4d] = torch.sign(x_grad[max_position_4d]).float()
            updated_x_i[opposite_channel_4d] = -1 *updated_x_i[max_position_4d] # if -1 in the space channel - it should be +1 in the blocked channel 
          # Clip the values to ensure they remain within the range of 0 or 1
        
        updated_x_i[:, channel, :, :] = torch.clamp(updated_x_i[:, channel, :, :], 0, 1)
        
        x_grad_abs[0,max_position] = -float('inf')
        
        if updated_x_i[max_position_4d] != x_i[max_position_4d]: # check if there was a meaningful change before further checks 
            updated_pos = max_position
            new_loss = compute_loss(updated_x_i, model, x_0=init_x, lambdas = lambdas,max_changes_dist=max_changes_dist )[0].item()
            
            if new_loss <= best_loss: 
                posits = []
                collision = False
                for ch in [2,3]: # could the change have led to any collisions -- the start pos and goal pos should not be replaced
                    try:
                        # coordinates = torch.where(updated_x_i[0, ch, :, :] == 1)
                        if updated_pos in posits:
                            posits.append(pos)
                            collision= True
                            print("Collision detected")
                    except:
                        collision= True
                
                if not collision:
                    grid_size, goal_positions, blocked_positions, start_pos,space_pos = decode_grid_design(updated_x_i[0].cpu())
                    # pdb.set_trace()
                    if len(black_list_pos) ==0: # block the fixed positions from being updated
                        black_list_pos.extend([(0*x_grad.shape[2]*x_grad.shape[2])+(pos[0]*x_grad.shape[2])+pos[1] for pos in goal_positions])
                        black_list_pos.extend([(1*x_grad.shape[2]*x_grad.shape[2])+(pos[0]*x_grad.shape[2])+pos[1] for pos in goal_positions])
                        black_list_pos.extend([(0*x_grad.shape[2]*x_grad.shape[2])+(start_pos[0]*x_grad.shape[2])+start_pos[1],
                                               (1*x_grad.shape[2]*x_grad.shape[2])+(start_pos[0]*x_grad.shape[2])+start_pos[1]])
                    
                    is_valid,new_shortest_path_lens = is_design_valid(grid_size, goal_positions, blocked_positions, start_pos)
                    if (np.array(new_shortest_path_lens)<=np.array(shortest_path_lens)).all() and is_valid:  # If a valid env is found, return True
                        found_x_i = True
                        break
                    else:
                        # print("Ivalid found, Updated pos",updated_pos)
                        invalid_x_is.append(updated_x_i.cpu())
                
        updated_x_i= x_i.clone()
        
    del x_grad_, x_grad_abs
    
    if found_x_i:
        return updated_x_i, channel, updated_pos,invalid_x_is, black_list_pos
    else:
        return updated_x_i, -1, -1,invalid_x_is, black_list_pos
    


def minimize_wcd(model, x, lambdas=[0.0,0.0],grid_size=6,max_iter=10, blocking_only= False, max_changes_dist = [3,5], sensitivity_analysis = False, noise_level = 0.0):
    alpha =0.00001
    x_i = x.clone()
    lowest_loss = compute_loss(x, model, x_0=x, lambdas = lambdas, max_changes_dist=max_changes_dist)[0].item()
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
    true_wcds =[compute_true_wcd_no_paths(x_i[0].cpu())]
    iters = [0]
    print("Pred:",wcds[-1]," True",true_wcds[-1])
    times = []
    shortest_path_lens = check_design_is_valid(x_i[0].cpu())[1]
    wcd_changes = []
    budgets = [i for i in range(1,max_iter,2)]
    cumulatiev_time = 0

    
    for i in range(1,max_iter+1):
        
        # Start the timer
        start_time = time.time()
     
        loss, wcd, x_grad = compute_gradients(x_i, model, x_0=x, lambdas = lambdas,max_changes_dist=max_changes_dist, sensitivity_analysis = sensitivity_analysis, noise_level = noise_level)
        # print("new loss",lowest_loss, model(x_i).item(),np.sum(decode_grid_design(x_i.cpu().squeeze(),return_map= True)!=decode_grid_design(x.cpu().squeeze(),return_map= True)))
        
        
        
        x_i,channel,updated_pos,invalid_x_is,black_list_pos = max_grad_update_single_channel(x_i,-1*x_grad 
                                                                                             , black_list_pos=black_list_pos,
                                                                                             model=model,     
                                                                                            best_loss=lowest_loss,grid_size=grid_size, 
                                                                                             shortest_path_lens =shortest_path_lens, 
                                                                                             
                                                                                             lambdas = lambdas, 
                                                                                             init_x = x,
                                                                                             max_changes_dist=max_changes_dist)
        black_list_pos.append(updated_pos)
        new_loss =compute_loss(x_i, model, x_0=x,lambdas = lambdas ,
                               max_changes_dist=max_changes_dist)[0].cpu().item()
        wcd = model(x_i.to(COMPUTE_DEVICE)).cpu()
        invalid_envs+=invalid_x_is


        if new_loss <lowest_loss:
            if check_design_is_valid(x_i[0].cpu()):  # If a valid env is found, return True
                pred_wcd_change = abs(wcd-model(best_x_i.to(COMPUTE_DEVICE)).cpu())
                best_x_i = x_i.clone()
                lowest_loss = new_loss
                
                if pred_wcd_change>=1:
                    no_progress =0
                else:
                    no_progress =+1
            else:
                no_progress += 1
            x_i = best_x_i
        else:
            no_progress += 1

       
        # End the timer
        end_time = time.time()
        
        changes = np.sum(decode_grid_design(best_x_i.cpu().squeeze(),
                                            return_map= True)!=decode_grid_design(x.cpu().squeeze(),
                                                                                  return_map= True))
        time_taken_ms = (end_time - start_time)
        cumulatiev_time += time_taken_ms
        
        if i%10==1:
            
            true_wcds.append(compute_true_wcd_no_paths(x_i[0].cpu()))
            if true_wcds[-1] is None:
                print("IS VALID ?", check_design_is_valid(x_i[0].cpu()),model(x_i.to(COMPUTE_DEVICE)).item())
                # plot_grid(decode_grid_design(x_i.cpu().squeeze(),return_map= True).tolist())
            wcds.append(wcd.item())
            iters.append(i)
            x_envs.append(x_i.cpu())
            print("i = ",i,"Pred:",wcd.item()," True",true_wcds[-1],"time taken",time_taken_ms)
            times.append(cumulatiev_time)
            wcd_changes.append(true_wcds[0]-true_wcds[-1])
        
        if updated_pos == -1:  
            break
        if no_progress >2:
            break
        
        true_wcds.append(compute_true_wcd_no_paths(best_x_i[0].cpu()))
        wcd_changes.append(true_wcds[0]-true_wcds[-1])
        
        
        
    
    true_wcds.append(compute_true_wcd_no_paths(best_x_i[0].cpu()))
    if true_wcds[-1] is None:
        print("IS VALID ?", check_design_is_valid(best_x_i[0].cpu()),model(best_x_i.to(COMPUTE_DEVICE)).item())
        # plot_grid(decode_grid_design(x_i.cpu().squeeze(),return_map= True).tolist())
    wcds.append(wcd)
    iters.append(i)
    x_envs.append(x_i)
    
    print("i = ",i,"Pred:",wcd.item()," True",true_wcds[-1],"total time taken",cumulatiev_time)
    times.append(cumulatiev_time)
    wcd_changes.append(true_wcds[0]-true_wcds[-1])
    
    
    return best_x_i,invalid_envs,wcds,true_wcds,x_envs, iters, cumulatiev_time, wcd_changes

def can_skip_file(env_file_name, expected_num_keys):
    """
    Checks if a JSON file exists and has the expected number of keys.

    Parameters:
    - env_file_name (str): Path to the JSON file.
    - expected_num_keys (int): The expected number of keys in the JSON file.

    Returns:
    - bool: True if the file exists and has the correct number of keys, False otherwise.
    """
    # Check if file exists
    if not os.path.exists(env_file_name):
        return False

    try:
        # Load the file content
        with open(env_file_name, 'r') as f:
            data = json.load(f)
            data = data['lambda_pairs']

        # Check number of keys
        if len(data) == expected_num_keys:
            return True
        else:
            return False

    except json.JSONDecodeError:
        return False

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
        default=10,  # Set the default value to 1
        help="Maximum grid size.",
    )
    
    parser.add_argument(
        "--max_iter",
        type=int,  # Ensure that the input is expected to be a int
        default=20,  # Set the default value to 1
        help="Maximum number of iterations.",
    )
    parser.add_argument(
        "--num_instances",
        type=int,  # Ensure that the input is expected to be a int
        default=5,  # Set the default value to 1
        help="spacing in the test dataset to use for the experiment",
    )
    
    
    parser.add_argument(
        "--experiment_type",
        type=str,  # Ensure that the input is expected to be a str
        default="ALL_MODS",  # Set the default value to 1
        # choices = ["ALL_MODS"],
        choices = ["BLOCKING_ONLY","ALL_MODS","BOTH_UNIFORM"],
        help="Either BLOCKING_ONLY or ALL_MODS for all modifications.",
    )
    
    parser.add_argument(
        "--ratio",
        type=str,  # Ensure that the input is expected to be a str
        default="1_1",  # Set the default value to 1:1
        # choices = ["ALL_MODS"],
        choices = ["1_1","1_2","1_3","3_1","1_5","5_1"],
        help="Either BLOCKING_ONLY or ALL_MODS for all modifications.",
    )

    parser.add_argument(
        "--wcd_pred_model_id",
        type=str,  # Ensure that the input is expected to be a str
        default="thunder-ascend-9224",  # Set the default value to 1
        help="Trained Model ID",
    )

    parser.add_argument(
        "--start_instance_index",
        type=int,  # Ensure that the input is expected to be a int
        default=0,  # Set the default value to 1
        help="ID of the first environment in the dataset",
    )

    parser.add_argument(
        "--interval",
        type=int,  # Ensure that the input is expected to be a int
        default=14,  # Set the default value to 1
        help="spacing in the test dataset to use for the experiment",
    )

    parser.add_argument(
        "--sensitivity_analysis",
        default=False,  # Set the default value to 1
        action="store_true", # store the value as True if the argument is present
        help="Whether to use sensitivity analysis",
    )

    parser.add_argument(
        "--noise_level",
        type=float,  # Ensure that the input is expected to be a float
        default=0.001,  # Set the default value to 1
        help="Noise level for sensitivity analysis",
    )
    
    # get_gpu_memory_usage(gpu_id=0)
    
    args = parser.parse_args()
    grid_size = args.grid_size
    experiment_type = args.experiment_type
    _label = "_best"
    num_instances = args.num_instances
    max_iter = args.max_iter
    ratio = args.ratio

    # Set up directories
    args.trained_model_dir = f"./models/wcd_prediction/grid{grid_size}/"
    if args.sensitivity_analysis:
        args.results_save_dir = f"./wcd_optim_results/ml-our-approach/grid{grid_size}/{args.wcd_pred_model_id}_sensitivity_analysis_{args.noise_level}_noise"
    else:
        args.results_save_dir = f"./wcd_optim_results/ml-our-approach/grid{grid_size}/{args.wcd_pred_model_id}"
    os.makedirs(args.results_save_dir, exist_ok=True)

    dataset_label = f"data/grid{grid_size}/model_training/dataset_{grid_size}{_label}.pkl"
    model_label = f"{args.trained_model_dir}/training_logs/{args.wcd_pred_model_id}/{args.wcd_pred_model_id}_model.pt"

    # Load dataset and model
    with open(dataset_label, "rb") as f:
        loaded_dataset = pickle.load(f)
    
    if args.start_instance_index>=len(loaded_dataset):
        print(f"index {args.start_instance_index} exceeds the number of environment instances")
        sys.exit()

    experiment_label = experiment_type + "_test"

    model = torch.load(model_label, map_location=torch.device(COMPUTE_DEVICE)).eval()

    # Initialize variables
    eval_changes = [1, 3, 5, 7, 9, 11, 13, 15, 17, 18]
    true_wcds_per_cost = []
    wcds_per_cost = []
    gammas = []
    times = []
    costs = []
    num_changes = []
    max_budgets = []
    all_wcd_changes = []
    realized_budgets = []
    start_from = args.start_instance_index

    # Experiment-specific configurations
    lambda_1_values = [0, 0.001, 0.002, 0.005, 0.007, 0.01, 0.02, 0.05, 0.07, 0.1, 0.2, 0.5, 0.7, 1.0, 2, 5, 7]

    if experiment_type == "BLOCKING_ONLY":
        lambda_2_values = [0.1, 0.5, 1, 10]
        blocking_rat = 1
        unblocking_rat = 0
    elif experiment_type == "ALL_MODS":
        lambda_2_values = lambda_1_values
        blocking_rat = int(ratio[0])
        unblocking_rat = int(ratio[2])
    else:
        if grid_size == 6:
            lambda_1_values += [0.0001, 0.0002, 0.005, 0.007, 10, 20, 100]
        lambda_1_values = np.insert(lambda_1_values, 0, 0)
        lambda_2_values = [0]
        blocking_rat = 1.05
        unblocking_rat = 1.0

    # Prepare storage paths
    args.data_storage_path = data_storage_path = f"{args.results_save_dir}/{experiment_label}/"
    args.data_storage_path_envs = f"{data_storage_path}envs"
    args.data_storage_path_lagrange = f"{data_storage_path}langrange_values"
    args.data_storage_path_model_enhancement = f"{data_storage_path}/model_training"
    # mode

    create_folder(data_storage_path)
    create_folder(args.data_storage_path_envs)
    create_folder(args.data_storage_path_model_enhancement)
    create_folder(args.data_storage_path_lagrange)

    max_budgets = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
    create_or_update_list_file(f"{data_storage_path}/max_budgets_{grid_size}_{experiment_label}.csv", [max_budgets])

    expected_num_keys = len(lambda_1_values) * len(lambda_1_values) # max number or langrage pairs
    
    # TODO limit the max index j

    for j in range(start_from, start_from+args.interval*args.num_instances,args.interval):
        
        if j>=len(loaded_dataset):
            print(f"index {j} exceeds the number of environment instances")
            continue

        env_file_name = f"{args.data_storage_path_lagrange}/env_{j}.json"


        if can_skip_file(env_file_name, expected_num_keys):
            print(f"{env_file_name} already exists; Skipping!")
            continue

        env_wcd_changes = []
        env_budgets = []
        env_times = []
        env_max_budget =[]
        max_budget = 40
        
        x, y = loaded_dataset[j]  # Get a specific data sample

        x = x.unsqueeze(0).float().cpu()
        
        env_dict = {
                "env_id": j,
                "lambda_pairs":[]
        }
        
        budget_buckets_realized = [[]]*len(max_budgets)
        budget_buckets_wcd_change = [-100]*len(max_budgets)
        budget_buckets_times = [0]*len(max_budgets)
        budget_x_envs = [x.cpu()]*len(max_budgets)
        budget_y_envs = [100]*len(max_budgets)
        
        for lambda_1 in  lambda_1_values: #0.01,0.05, 0.08,0.1,0.2,0.36
            for lambda_2 in lambda_2_values:
                print("Environment; ",j, "Cost: ",[lambda_1, lambda_2])
                
                best_wcd = []
                invalid_envs_collection =[]
                true_wcds = []
                max_changes_dist=np.round([(blocking_rat*max_budget)/(unblocking_rat+blocking_rat),
                                            (unblocking_rat*max_budget)/(unblocking_rat+blocking_rat) ]).tolist()

                max_iter = max_budget
                best_x_i, invalid_envs,wcds,true_wcds,x_envs,iters, time_taken, wcd_changes = minimize_wcd(
                    model, x, 
                    lambdas=[lambda_1, lambda_2],
                    max_changes_dist=max_changes_dist if experiment_type != "BOTH_UNIFORM" else [-1,-1] ,
                    grid_size =grid_size, 
                    max_iter = max_iter, 
                    blocking_only = False,
                    sensitivity_analysis = args.sensitivity_analysis,
                    noise_level = args.noise_level
                )
                env_times.append(time_taken)
                env_wcd_changes.append(wcd_changes[-1])
                wcds_per_cost.append(wcds)
                true_wcds_per_cost.append(true_wcds)

                if len(x_envs) ==1:
                        continue

                costs.append([lambda_1, lambda_2])
                n_changes = np.sum(decode_grid_design(best_x_i.cpu().squeeze(),
                                                        return_map= True)!=decode_grid_design(x.cpu().squeeze(),        
                                                                                            return_map= True))
                # pdb.set_trace()
                x_changes = best_x_i.cpu()[:, 1, :, :]-x.cpu()[:, 1, :, :]
                blockings = (x_changes==1).sum(axis=(1, 2))
                removals = (x_changes==-1).sum(axis=(1, 2))
                num_changes.append([blockings.item(),removals.item()])
                wcd_change = wcd_changes[-1]
                
                # if abs(wcds[-1]-true_wcds[-1])>1: #
                #     update_or_create_dataset(f"{args.data_storage_path_model_enhancement}/simulated_valids_final{grid_size}_{1 if 'ALL_MODS' in experiment_label else 0}.pkl", [x_envs[-1]], [true_wcds[-1]])
                
                print("Final X:, WCD = ",model(best_x_i.to(COMPUTE_DEVICE)).item(),"True",true_wcds[-1],
                        "n_changes",n_changes, [blockings.item(), removals.item()] ,"Bugdet",max_budget, "Time taken",time_taken)
                
                env_dict["lambda_pairs"].append({
                    "lambdas": [lambda_1, lambda_2],
                    "wcd_change": int(wcd_change),
                    "num_changes": [blockings.item(), removals.item()],
                    "time_taken": time_taken
                })
            
            with open(env_file_name, "w") as json_file:
                json.dump(env_dict, json_file, indent=4)

        budget_x_envs.insert(0,x.cpu()) # the first entry should be the initial environment
        budget_y_envs.insert(0,y.cpu()) # the first entry should be the initial WCD
        
        update_or_create_dataset(f"{args.data_storage_path_envs}/envs_{j}_{grid_size}_{experiment_type}.pkl", 
                                    budget_x_envs, budget_y_envs) # store the final environments   
                        
        
        # pdb.set_trace()
        times.append(budget_buckets_times)
        all_wcd_changes.append(budget_buckets_wcd_change)
        realized_budgets.append(budget_buckets_realized)
        
        create_or_update_list_file(f"{data_storage_path}/times_{grid_size}_{experiment_label}.csv",times)
        create_or_update_list_file(f"{data_storage_path}/wcd_change_{grid_size}_{experiment_label}.csv",all_wcd_changes)
        create_or_update_list_file(f"{data_storage_path}/budgets_{grid_size}_{experiment_label}.csv",realized_budgets)
