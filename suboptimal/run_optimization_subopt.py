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



def compute_loss(x,model, x_0, lambdas =[0,0],max_changes_dist = [3, 5]):
    
    # Forward pass
    output = model(x)
    # Define a scalar loss function
    lambda_1,lambda_2 = lambdas[0],lambdas[1]
    wcd = output.mean()
    
    x_changes = x[:, 1, :, :]-x_0[:, 1, :, :]
    

    blockings = torch.sum(F.softplus(x_changes))
    removals = torch.sum(F.softplus(torch.abs(-x_changes)))
    
    if max_changes_dist == [-1,-1]: # no limits
        penalty_term = lambda_1*(blockings+removals)

        sim_loss = penalty_term
    else:
        
        penalty_term_1 = lambda_1 * blockings
        penalty_term_2 = lambda_2 * removals

        sim_loss = penalty_term_1 + penalty_term_2

    loss =(wcd +sim_loss)
    # print("loss",loss.item(),"WCD",wcd.item(),"Removals",torch.sum(F.softplus(-x_changes)).item(), "Blockings:",torch.sum(F.softplus(x_changes)).item())
    return loss,wcd

    
def compute_gradients(x,model, x_0, lambdas =[0,0],max_changes_dist = [3, 5]):
    x.requires_grad = True
    # print("loss:",loss.item(), "WCD :", wcd.item(), "regukarizer:", sim_loss.item(), "penalty:", lambda*sim_loss.item())
    loss,wcd = compute_loss(x,model, x_0, lambdas =lambdas,max_changes_dist = max_changes_dist)
    loss.backward()
    
    # Access gradients for x
    gradients = x.grad
    x.requires_grad = False
    return loss.item(),wcd.item(),gradients
    



def max_grad_update_single_channel(x_i, x_grad, black_list_pos =[],model=None, best_loss = None,grid_size=19,shortest_path_lens=None, init_x = None, lambdas = [0,0], max_changes_dist=[3,5] , K=4):
    updated_x_i = x_i.clone()
    
    # Find the channel with the maximum gradient
    import pdb
    
    x_grad_ = x_grad.view((x_grad.size(0), x_grad.size(1),-1))[:, 1, :] # channels  0,2,3 cannot change - only blocking channel (=1)
    x_grad_abs = x_grad[:, 1, :].abs().view(1, -1)

    updated_pos = -1 # default if no change is made
    
    for pos in black_list_pos:# disqualify these
        x_grad_abs[0,pos] = -float('inf')
        
    possible_x_is =[]
    
    invalid_x_is =[]
    
    found_x_i = False
    for i in range(x_grad_abs.shape[1]):
        max_position = torch.argmax(x_grad_abs)
        # Convert the flattened index to 4D coordinates
        
        channel_dim = 1 #only the blocking channel
        max_position_remainder = max_position % (x_grad.shape[2] * x_grad.shape[3])
        height_dim = max_position_remainder // x_grad.shape[3]
        width_dim = max_position_remainder % x_grad.shape[3]
    
        max_position_4d = (
            0,  # Batch dimension
            channel_dim,  # Channel dimension
            height_dim,  # Height dimension
            width_dim  # Width dimension
        )
        
        channel = channel_dim
        

        updated_x_i[max_position_4d] = torch.sign(x_grad[max_position_4d]).float()
               
        
        updated_x_i[:, channel, :, :] = torch.clamp(updated_x_i[:, channel, :, :], 0, 1)
        
        x_grad_abs[0,max_position] = -float('inf')
        
        if updated_x_i[max_position_4d] != x_i[max_position_4d]: # check if there was a meaningful change before further checks 
            updated_pos = max_position
            new_loss = compute_loss(updated_x_i, model, x_0=init_x, lambdas = lambdas,max_changes_dist=max_changes_dist )[0].item()
            
            if new_loss <= best_loss: # is the predicted WCD reducing?
                posits = []
                collision = False
                for ch in [0,3]: # could the change have led to any collisions -- the start pos and goal pos should not be replaced
                        coordinates = torch.where(updated_x_i[0, ch, :, :] == 1)
                        # Convert coordinates to a list of tuples
                        coord_list = list(zip(coordinates[0].tolist(), coordinates[1].tolist()))

                
                        if updated_pos in coord_list:
                            collision= True
                            print("Collision detected")
                
                if not collision:
                    
                    if len(black_list_pos) ==0: # block the fixed positions from being updated
                        x_encoded = decode_mdp_design(x_i.cpu(),K=K)
                        goal_positions = x_encoded["goal_positions"]
                        start_pos = x_encoded["start_pos"]
                        
                        black_list_pos.extend([(pos[0]*x_grad.shape[2])+pos[1] for pos in goal_positions])
                        black_list_pos.extend([(start_pos[0]*x_grad.shape[2])+start_pos[1]])
                    
                    is_valid,new_shortest_path_lens = is_x_valid(updated_x_i.cpu(),K=K)
                    if is_valid:  # If a valid env is found, return True
                        found_x_i = True
                        break
                    else:
                        # print("Ivalid found, Updated pos",updated_pos)
                        invalid_x_is.append(updated_x_i)
                
        updated_x_i= x_i.clone()
        
                
    if found_x_i:
        return updated_x_i, channel, updated_pos,invalid_x_is, black_list_pos
    else:
        return updated_x_i, -1, -1,invalid_x_is, black_list_pos
    


def minimize_wcd(model, x, lambdas=[0.0,0.0],grid_size=6,max_iter=10, K= 4, max_changes_dist = [3,5],init_true_wcd = 0):
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
    wcds = [model(x_i).item()]
    x_envs = [x_i]
    true_wcds =[init_true_wcd]
    iters = [0]
    print("Pred:",wcds[-1]," True",true_wcds[-1])
    times = []
    shortest_path_lens = check_design_is_valid(x_i[0].cpu(),K=K)[1]
    wcd_changes = []
    budgets = [i for i in range(1,max_iter,2)]
    cumulatiev_time = 0.0000001
    
    if true_wcds[0]==0:
        return best_x_i,invalid_envs,wcds,true_wcds,x_envs, iters, cumulatiev_time,[0]
    
    
    for i in range(1,max_iter+1):
        
        # Start the timer
        start_time = time.time()
        
        loss, wcd, x_grad = compute_gradients(x_i, model, x_0=x, lambdas = lambdas,max_changes_dist=max_changes_dist )
        x_i,channel,updated_pos,invalid_x_is,black_list_pos = max_grad_update_single_channel(x_i,-1*x_grad, 
                                                                                             black_list_pos=black_list_pos,model=model
                                                                              ,best_loss=lowest_loss,grid_size=grid_size, 
                                                                                             shortest_path_lens =shortest_path_lens, 
                                                                               lambdas = lambdas, init_x = x,max_changes_dist=max_changes_dist,
                                                                                            K=K)
        
        black_list_pos.append(updated_pos)
        new_loss =compute_loss(x_i, model, x_0=x,lambdas = lambdas ,max_changes_dist=max_changes_dist)[0].item()
        wcd = model(x_i)
        invalid_envs+=invalid_x_is


        if new_loss <lowest_loss:
            if check_design_is_valid(x_i[0].cpu(), K=K):  # If a valid env is found, return True
                # print("**** BEST ***")
                best_x_i = x_i.clone()
                lowest_loss = new_loss
                no_progress =0
            else:
                no_progress += 1
            x_i = best_x_i
        else:
            no_progress += 1
        # End the timer
        end_time = time.time()
        
        changes = np.sum(decode_mdp_design(best_x_i.cpu().squeeze(),
                                           return_grid= True, K=K)[1]!=decode_mdp_design(x.cpu().squeeze(),
                                                                                         return_grid= True, K=K)[1])
        # Calculate the time taken in milliseconds
        time_taken_ms = (end_time - start_time)
        cumulatiev_time += time_taken_ms
        
        if i%30==0:
            true_wcds.append(compute_true_wcd(x_i[0].cpu(),K=K))
            if true_wcds[-1] is None:
                print("IS VALID ?", check_design_is_valid(x_i[0].cpu(), K=K),model(x_i).item())
                # plot_grid(decode_grid_design(x_i.cpu().squeeze(),return_map= True).tolist())
            wcds.append(wcd.item())
            iters.append(i)
            x_envs.append(x_i)
            print("i = ",i,"Pred:",wcd.item()," True",true_wcds[-1],"time taken",time_taken_ms)
            times.append(cumulatiev_time)
            wcd_changes.append(true_wcds[0]-true_wcds[-1])
        
        if updated_pos == -1:  
            break
        if no_progress >10:
            break
        if wcd <= 0.1:
            break
        
    true_wcds.append(compute_true_wcd(best_x_i[0].cpu(),K=K))
    wcds.append(model(best_x_i).item())
    wcd_changes.append(true_wcds[0]-true_wcds[-1])
    
        
   
    return best_x_i,invalid_envs,wcds,true_wcds,x_envs, iters, cumulatiev_time, wcd_changes


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
        "--start_index",
        type=int,  # Ensure that the input is expected to be a float
        default=0,  # Set the default value to 1
        help="Starting index for the number of instances",
    )
    
    
    parser.add_argument(
        "--experiment_type",
        type=str,  # Ensure that the input is expected to be a str
        default="ALL_MODS",  # Set the default value to 1
        # choices = ["ALL_MODS"],
        choices = ["BLOCKING_ONLY","ALL_MODS","BOTH_UNIFORM"],
        help="Either BLOCKING_ONLY or ALL_MODS for all modifications.",
    )

    
    args = parser.parse_args()
    grid_size = args.grid_size
    experiment_label = "test"
    experiment_type = args.experiment_type
    _label = "_best"
    K = args.K
    
    max_iter = args.max_iter
    
    dataset_label = f"data/grid{grid_size}/model_training/dataset_{grid_size}_K{K}{_label}.pkl"
    model_label = f"models/wcd_nn_model_{grid_size}_K{K}{_label}.pt"#{_label}
   

    with open(dataset_label, "rb") as f:
        loaded_dataset = pickle.load(f)
        
    experiment_label=experiment_type+ "_"+experiment_label

    device ="cuda:0"
    model = torch.load(model_label)
    model = model.to(device).eval()

    true_wcds_per_cost=[]
    wcds_per_cost = []
    gammas = []
    times = []
    costs = []
    num_changes = []
    max_budgets =  [1,3,5,7,9,11,13,15,17,19]
    
    times = []
    all_wcd_changes = []
    realized_budgets =[]
    
    gammas = []
    true_wcds_list = []
    pred_wcds_list = []
    
    if experiment_type ==  "ALL_MODS" or experiment_type == "BLOCKING_ONLY": #ALL Modifications are allowed
        lambda_1_values = [0,0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1.0,2,5]
        lambda_2_values = lambda_1_values
        blocking_rat = 3
        unblocking_rat = 5
    else:
        lambda_1_values = [0,0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1.0,2,5]
        lambda_2_values = [0]
        blocking_rat = 1
        unblocking_rat = 1
    
    
    # data_storage_path =f"data/grid{grid_size}/{experiment_label}/"
    data_storage_path =f"data/grid{grid_size}/K{K}/{experiment_label}/"
    create_folder(data_storage_path)
        

    create_folder(data_storage_path+"langrange_values")
    create_folder(data_storage_path)
    
    
    create_or_update_list_file(f"{data_storage_path}/max_budgets_{grid_size}_{experiment_label}.csv",[max_budgets])
    
    max_index = len(loaded_dataset)
    interval = 1
    max_index = np.min([args.start_index+(interval*2000),len(loaded_dataset)])
    
    print("Interval : ",interval)
    
    for k in range(args.start_index, len(loaded_dataset),interval):
        env_file_path = f"./data/grid{grid_size}/K{K}/{experiment_label}/langrange_values/env_{k}.json"
        if os.path.exists(env_file_path):
             with open(env_file_path, "r") as json_file:
                data = json.load(json_file)
                # print(json_file)
                if len(data["lambda_pairs"])>=256:
                    continue #this is already ran
        
        try:
            env_wcd_changes = []
            env_budgets = []
            env_times = []
            env_max_budget =[]
            
            x, y = loaded_dataset[k]  # Get a specific data sample

            x = x.unsqueeze(0).float().cuda()
            max_budget = 25
            
            env_dict = {
                    "env_id": k,
                    "lambda_pairs":[]
            }
            time_start = time.time()
            init_true_wcd = compute_true_wcd(x[0].cpu(),K=K)
            time_to_compute = time.time()-time_start
            for lambda_1 in  lambda_1_values: #0.01,0.05, 0.08,0.1,0.2,0.36
                for lambda_2 in lambda_2_values:
                    print("Environment; ",k, "Cost: ",[lambda_1, lambda_2])
                    print("Original X:, pred WCD = ",model(x).item(), "true WCD",init_true_wcd,"time:",time_to_compute)
                    # plot_grid(decode_grid_design(x.cpu().squeeze(),return_map= True).tolist())
                    best_wcd = []
                    invalid_envs_collection =[]
                    true_wcds = []
                    max_changes_dist=np.round([(blocking_rat*max_budget)/(unblocking_rat+blocking_rat),
                                               (unblocking_rat*max_budget)/(unblocking_rat+blocking_rat) ]).tolist()

                    max_iter = max_budget
                    best_x_i, invalid_envs,wcds,true_wcds,x_envs,iters, time_taken, wcd_changes = minimize_wcd(model, 
                                                                                                               x, 
                                                                                                               
                                                                    lambdas=[lambda_1, lambda_2], 
                                        max_changes_dist=max_changes_dist if experiment_type != "BOTH_UNIFORM" else [-1,-1]
                                                                                                               ,           
                                        grid_size =grid_size, max_iter = max_iter, K = K, init_true_wcd= init_true_wcd
                                                                                                )
                                                                                                 # blocking_only = True if experiment_type=="BLOCKING_ONLY" else False )
                    true_wcds_list.extend(true_wcds)
                    pred_wcds_list.extend(wcds)
                    env_times.append(time_taken)
                    env_wcd_changes.append(wcd_changes[-1])
                    wcds_per_cost.append(wcds)
                    true_wcds_per_cost.append(true_wcds)


                    costs.append([lambda_1, lambda_2])
                    n_changes = np.sum(decode_mdp_design(best_x_i.cpu().squeeze(),
                                                         return_grid= True, K=K)[1]!=decode_mdp_design(x.cpu().squeeze(),
                                                                                                       return_grid= True,
                                                                                                       K=K)[1])
                    # pdb.set_trace()
                    x_changes = best_x_i.cpu()[:, 1, :, :]-x.cpu()[:, 1, :, :]
                    blockings = (x_changes==1).sum(axis=(1, 2))
                    removals = (x_changes==-1).sum(axis=(1, 2))


                    num_changes.append([blockings.item(),removals.item()])
                    # env_max_budget.append(max_changes_dist)
                    env_budgets.append([blockings.item(),removals.item()])
                    
                    wcd_change = wcd_changes[-1]
                    
                    # if abs(wcds[-1]-true_wcds[-1])>1:
                    #     update_or_create_dataset(f"data/grid{grid_size}/model_training/simulated_valids_final{grid_size}_K{K}.pkl", 
                    #                              [x_envs[-1]], [true_wcds[-1]])
                    
                    print("Final X:, WCD = ",model(best_x_i).item(),"True",true_wcds[-1],"n_changes",n_changes,
                          "Bugdet",max_budget,"Constraint",[blockings.item(),
                                                            removals.item()]<=max_changes_dist, "Time taken",time_taken)

#                    
                    env_dict["lambda_pairs"].append({
                        "lambdas": [lambda_1, lambda_2],
                        "wcd_change": wcd_change,
                        "num_changes": [blockings.item(), removals.item()],
                        "time_taken": time_taken
                    })
            
                with open(f"./data/grid{grid_size}/K{K}/{experiment_label}/langrange_values/env_{k}.json", "w") as json_file:
                    json.dump(env_dict, json_file, indent=4)
        
        except Exception as e:
                print("Exception details:")
                traceback.print_exc()

    # plot_combined(wcds_per_cost,true_wcds_per_cost,gammas)
     