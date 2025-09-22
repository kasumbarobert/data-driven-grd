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
import json

human_model = torch.load('models/human_model_grid6.pt', map_location=torch.device('cpu'))
human_model.eval()

def compute_loss(x,model, x_0, lambdas =[0,0],max_changes_dist = [3, 5]):
    
    # Forward pass
    output = model(x)
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
        
        # penalty_term_1 = lambda_1 * torch.max(torch.tensor(0), blockings)**2
        # penalty_term_2 = lambda_2 * torch.max(torch.tensor(0), removals)**2
        
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



def max_grad_update_single_channel(x_i, x_grad, black_list_pos =[],model=None, best_loss = None,grid_size=19,shortest_path_lens=None, init_x = None, lambdas = [0,0], max_changes_dist=[3,5] ):
    updated_x_i = x_i.clone()
    
    # Find the channel with the maximum gradient
    import pdb

    x_grad_ = x_grad.view((x_grad.size(0), x_grad.size(1),-1))[:, 0:2, :] # channels  2,3 cannot change
    x_grad_abs = x_grad[:, 0:2, :].abs().view(1, -1)
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
            
            if new_loss <= best_loss: # is the predicted WCD reducing?
                # print("LOSS:",new_loss,model(updated_x_i).item(),np.sum(decode_grid_design(updated_x_i.cpu().squeeze(),return_map= True)!=decode_grid_design(init_x.cpu().squeeze(),return_map= True)))
                posits = []
                collision = False
                for ch in [2,3]: # could the change have led to any collisions -- the start pos and goal pos should not be replaced
                    try:
                        coordinates = torch.where(updated_x_i[0, ch, :, :] == 1)
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
    


def minimize_wcd(model, x, lambdas=[0.0,0.0],grid_size=6,max_iter=10, max_changes_dist = [3,5],search_depth = 19):
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
    true_wcds =[compute_humal_model_wcd(x_i[0].cpu(),model =human_model,search_depth=search_depth)]
    iters = [0]
    print("Pred:",wcds[-1]," True",true_wcds[-1])
    times = []
    shortest_path_lens = check_design_is_valid(x_i[0].cpu(),human_model=human_model)[1]
    wcd_changes = []
    budgets = [i for i in range(1,max_iter,2)]
    cumulatiev_time = 0
    
    for i in range(1,max_iter+1):
        
        # Start the timer
        start_time = time.time()
        
        loss, wcd, x_grad = compute_gradients(x_i, model, x_0=x, lambdas = lambdas,max_changes_dist=max_changes_dist )
        # print("new loss",lowest_loss, model(x_i).item(),np.sum(decode_grid_design(x_i.cpu().squeeze(),return_map= True)!=decode_grid_design(x.cpu().squeeze(),return_map= True)))
        
        x_i,channel,updated_pos,invalid_x_is,black_list_pos = max_grad_update_single_channel(x_i,-1*x_grad 
                                                                                             , black_list_pos=black_list_pos,model=model,     
                                                                                            best_loss=lowest_loss,grid_size=grid_size, 
                                                                                             shortest_path_lens =shortest_path_lens, 
                                                                                               lambdas = lambdas, 
                                                                                             init_x = x,max_changes_dist=max_changes_dist)
        
        black_list_pos.append(updated_pos)
        new_loss =compute_loss(x_i, model, x_0=x,lambdas = lambdas ,max_changes_dist=max_changes_dist)[0].item()
        wcd = model(x_i)
        invalid_envs+=invalid_x_is


        if new_loss <lowest_loss:
            if check_design_is_valid(x_i[0].cpu(),human_model=human_model):  # If a valid env is found, return True
                # print("**** BEST ***")
                old_wcd = model(best_x_i)
                best_x_i = x_i.clone()
                lowest_loss = new_loss
                
                
                if abs(old_wcd-model(x_i))>0.5:
                    no_progress =0
                else:
                    no_progress +=1
            else:
                no_progress += 1
            x_i = best_x_i
        else:
            no_progress += 1

       
        # End the timer
        end_time = time.time()
        
        changes = np.sum(decode_grid_design(best_x_i.cpu().squeeze(),return_map= True)!=decode_grid_design(x.cpu().squeeze(),return_map= True))
        # Calculate the time taken in milliseconds
        time_taken_ms = (end_time - start_time)
        cumulatiev_time += time_taken_ms
        
        
            
        
        if i%10==1:
            
            true_wcds.append(compute_humal_model_wcd(x_i[0].cpu(),model =human_model,search_depth =search_depth))
            if true_wcds[-1] is None:
                print("IS VALID ?", check_design_is_valid(x_i[0].cpu(),human_model=human_model),model(x_i).item())
                continue
                # plot_grid(decode_grid_design(x_i.cpu().squeeze(),return_map= True).tolist())
            wcds.append(wcd)
            iters.append(i)
            x_envs.append(x_i)
            print("i = ",i,"Pred:",wcd.item()," True",true_wcds[-1],"time taken",time_taken_ms)
            times.append(cumulatiev_time)
            
            wcd_changes.append(true_wcds[0]-true_wcds[-1])
        
        if updated_pos == -1:  
            break
        if no_progress >2:
            break
        
        true_wcds.append(compute_humal_model_wcd(best_x_i[0].cpu(),model =human_model,search_depth=search_depth))
        
        if true_wcds[-1] is None:
            wcd_changes.append(0)
        else:
            wcd_changes.append(true_wcds[0]-true_wcds[-1])
        
    
    true_wcds.append(compute_humal_model_wcd(best_x_i[0].cpu(),model =human_model))
    
    if true_wcds[-1] is None:
        print("IS VALID ?", check_design_is_valid(best_x_i[0].cpu(),human_model=human_model),model(best_x_i).item())
        # plot_grid(decode_grid_design(x_i.cpu().squeeze(),return_map= True).tolist())
    wcds.append(wcd)
    iters.append(i)
    x_envs.append(x_i)
    
    print("i = ",i,"Pred:",wcd.item()," True",true_wcds[-1],"time taken",time_taken_ms)
    times.append(cumulatiev_time)
    if true_wcds[-1] is None:
         wcd_changes.append(0)
    else:
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
        "--cost",
        type=float,  # Ensure that the input is expected to be a float
        default=0,  # Set the default value to 0
        help="Cost parameter for the simulation. Default is 0.",
    )
    
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
        "--start_index",
        type=int,  # Ensure that the input is expected to be a int
        default=0,  # Set the default value to 1
        help="Starting index for the environments",
    )
    
    parser.add_argument(
        "--num_instances",
        type=int,  # Ensure that the input is expected to be a int
        default=400,  # Set the default value to 1
        help="spacing in the test dataset to use for the experiment",
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
        default="ALL_MODS",  # Set the default value to 1
        # choices = ["ALL_MODS"],
        choices = ["BLOCKING_ONLY","ALL_MODS","BOTH_UNIFORM"],
        help="Either BLOCKING_ONLY or ALL_MODS for all modifications.",
    )
    
    parser.add_argument(
        "--assumed_behavior",
        type=str,  # Ensure that the input is expected to be a str
        default="HUMAN",  # Set the default value to 1
        # choices = ["ALL_MODS"],
        choices = ["OPTIMAL","HUMAN"]
    )

    
    args = parser.parse_args()
    cost = args.cost
    grid_size = args.grid_size
    experiment_label = args.experiment_label
    experiment_type = args.experiment_type
    _label = "_best"
    num_instances = args.num_instances
    
    max_iter = args.max_iter
    
    search_depth = 10 if grid_size ==6 else 19

    dataset_label = f"data/grid{grid_size}/model_training/dataset_{grid_size}_may10.pkl"
    if args.assumed_behavior=="OPTIMAL":
        model_label = f"models/wcd_nn_model_{grid_size}{_label}_optimal.pt"
    else:
        model_label = f"models/wcd_nn_model_{grid_size}_human_may10.pt"

    with open(dataset_label, "rb") as f:
        loaded_dataset = pickle.load(f)
        
    experiment_label=experiment_type+ "_"+experiment_label

    human_model = torch.load(f'models/human_model_grid{grid_size}.pt', map_location=torch.device('cpu'))
    device ="cuda:0"
    model = torch.load(model_label,map_location=torch.device('cpu'))
    model = model.to(device).eval()

    true_wcds_per_cost=[]
    wcds_per_cost = []
    gammas = []
    times = []
    costs = []
    num_changes = []
    max_budgets = []
    
    eval_changes = [1,3,5,7,9,11,13,15,17,18]
    
    times = []
    all_wcd_changes = []
    realized_budgets =[]
    
    if experiment_type == "BLOCKING_ONLY":
        lambda_1_values = [0.001,0.005,0.01,0.05,0.1,0.2,0.5,1.0,5]
        lambda_2_values = [0.1,0.5,1,10,100]
        blocking_rat = 1
        unblocking_rat = 0
    elif experiment_type ==  "ALL_MODS": #ALL Modifications are allowed
        lambda_1_values = [0,0.005,0.01,0.05,0.1,0.20,0.5,1.0,5]
        lambda_2_values = [0,0.005,0.01,0.05,0.1,0.20,0.5,1.0,5]
        blocking_rat = 3
        unblocking_rat = 5
    else:
        lambda_1_values = [0,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1.0,2,5]
        lambda_2_values = [0]
        blocking_rat = 1.05
        unblocking_rat = 1.0
    
    data_storage_path =f"data/grid{grid_size}/{args.assumed_behavior}/"
    create_folder(data_storage_path)
    create_folder(data_storage_path+"/langrange_values")
    
    max_budgets = [1,3,5,7,9,11,13,15,17,19,21,23]
    create_or_update_list_file(f"{data_storage_path}/max_budgets_{grid_size}_{experiment_label}.csv",[max_budgets])
    
    max_index = len(loaded_dataset)
    interval= len(loaded_dataset)//num_instances
    print("Interval ", interval)
    max_index = np.min([args.start_index+(interval*100),len(loaded_dataset)])
    
    for j in range(args.start_index, max_index,interval):
            env_wcd_changes = []
            env_budgets = []
            env_times = []
            env_max_budget =[]
            max_budget = 25
            
            budget_buckets_realized = [[]]*len(max_budgets)
            budget_buckets_wcd_change = [-100]*len(max_budgets)
            budget_buckets_times = [0]*len(max_budgets)
            x, y = loaded_dataset[j]  # Get a specific data sample
            x = x.unsqueeze(0).float().cuda()
            env_dict = {
                    "env_id": j,
                    "lambda_pairs":[]
            }
            for lambda_1 in  lambda_1_values: #0.01,0.05, 0.08,0.1,0.2,0.36
                for lambda_2 in lambda_2_values:
                    try:
                        print("Environment; ",j, "Cost: ",[lambda_1, lambda_2])
                        

                        print("Original X:, WCD = ",model(x).item())
                        # plot_grid(decode_grid_design(x.cpu().squeeze(),return_map= True).tolist())
                        best_wcd = []
                        invalid_envs_collection =[]
                        true_wcds = []
                        max_changes_dist=np.round([(blocking_rat*max_budget)/(unblocking_rat+blocking_rat)
                                                   ,(unblocking_rat*max_budget)/(unblocking_rat+blocking_rat) ]).tolist()

                        max_iter = max_budget
                        best_x_i, invalid_envs,wcds,true_wcds,x_envs,iters, time_taken, wcd_changes = minimize_wcd(model, 
                                                                                                                   x, 
                                                                                                lambdas=[lambda_1, lambda_2],
                                                            max_changes_dist=max_changes_dist if experiment_type != "BOTH_UNIFORM" else [-1,-1] ,
                                                            grid_size =grid_size, max_iter = max_iter, search_depth = search_depth
                                                                                                    )
                        env_times.append(time_taken)
                        env_wcd_changes.append(wcd_changes[-1])
                        wcds_per_cost.append(wcds)
                        true_wcds_per_cost.append(true_wcds)

                        if len(x_envs) ==1:
                                continue

                        costs.append([lambda_1, lambda_2])
                        n_changes = np.sum(decode_grid_design(best_x_i.cpu().squeeze(),
                                                              return_map= True)!=decode_grid_design(x.cpu().squeeze()
                                                            ,return_map= True))
                        # pdb.set_trace()
                        x_changes = best_x_i.cpu()[:, 1, :, :]-x.cpu()[:, 1, :, :]
                        blockings = (x_changes==1).sum(axis=(1, 2))
                        removals = (x_changes==-1).sum(axis=(1, 2))
                        num_changes.append([blockings.item(),removals.item()])
                        wcd_change = wcd_changes[-1]

                        print("Final X:, WCD = ",model(best_x_i).item(),"True",true_wcds[-1],"n_changes",
                              n_changes,"Bugdet",max_budget, "Time taken",time_taken)

                        if abs(true_wcds[-1]-model(x_envs[-1]))>=1:
                            update_or_create_dataset(f"data/grid{grid_size}/model_training/simulated_valids_final{grid_size}.pkl", 
                                                     [x_envs[-1]], [true_wcds[-1]])
                        
                        if true_wcds[-1]>=0:
                            env_dict["lambda_pairs"].append({
                                "lambdas": [lambda_1,lambda_2],
                                "wcd_change": wcd_change,
                                "num_changes": num_changes[-1],
                                "time_taken": time_taken
                            })

                            with open(f"{data_storage_path}/langrange_values/env_{j}.json", "w") as json_file:
                                json.dump(env_dict, json_file, indent=4)
                        
                    except Exception as e:
                        print("Exception details:")
                        traceback.print_exc()

     