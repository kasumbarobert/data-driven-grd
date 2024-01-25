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



def max_grad_update_single_channel(x_i, x_grad, black_list_pos =[],model=None, best_loss = None,grid_size=19,shortest_path_lens=None, blocking_only = False, init_x = None, lambdas = [0,0], max_changes_dist=[3,5] ):
    updated_x_i = x_i.clone()
    
    # Find the channel with the maximum gradient
    import pdb
    # pdb.set_trace()
    if blocking_only: # blocking actions only allowed - this means only the blocked channel is update
        x_grad_ = x_grad.view((x_grad.size(0), x_grad.size(1),-1))[:, 1, :] # channels  0,2,3 cannot change - only blocking channel (=1)
        x_grad_abs = x_grad[:, 1, :].abs().view(1, -1)
    else:
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
        if blocking_only: 
            channel_dim = 1 #only the blocking channel
        else:
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
        
        if blocking_only: #only additions are allowed
            if torch.sign(x_grad[max_position_4d]).float() ==1:
                updated_x_i[max_position_4d] = torch.sign(x_grad[max_position_4d]).float()
                updated_x_i[opposite_channel_4d] = -1 *updated_x_i[max_position_4d] # if -1 in the space channel - it should be +1 in the blocked channel 
        
        else:
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
                    if (np.array(new_shortest_path_lens)<=np.array(shortest_path_lens)).all() and is_valid:  # If a valid env is found, return True
                        found_x_i = True
                        break
                    else:
                        invalid_x_is.append(updated_x_i)
                
        updated_x_i= x_i.clone()
        
                
    if found_x_i:
        return updated_x_i, channel, updated_pos,invalid_x_is, black_list_pos
    else:
        return updated_x_i, -1, -1,invalid_x_is, black_list_pos
    


def minimize_wcd(model, x, lambdas=[0.0,0.0],grid_size=6,max_iter=10, blocking_only= False, max_changes_dist = [3,5]):
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
    true_wcds =[compute_true_wcd(x_i[0].cpu())]
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
        
        loss, wcd, x_grad = compute_gradients(x_i, model, x_0=x, lambdas = lambdas,max_changes_dist=max_changes_dist )
        # print("new loss",lowest_loss, model(x_i).item(),np.sum(decode_grid_design(x_i.cpu().squeeze(),return_map= True)!=decode_grid_design(x.cpu().squeeze(),return_map= True)))
        
        x_i,channel,updated_pos,invalid_x_is,black_list_pos = max_grad_update_single_channel(x_i,-1*x_grad 
                                                                                             , black_list_pos=black_list_pos,model=model,     
                                                                                            best_loss=lowest_loss,grid_size=grid_size, 
                                                                                             shortest_path_lens =shortest_path_lens, 
                                                                                              blocking_only = blocking_only, lambdas = lambdas, 
                                                                                             init_x = x,max_changes_dist=max_changes_dist)
        
        black_list_pos.append(updated_pos)
        new_loss =compute_loss(x_i, model, x_0=x,lambdas = lambdas ,max_changes_dist=max_changes_dist)[0].item()
        wcd = model(x_i)
        invalid_envs+=invalid_x_is


        if new_loss <lowest_loss:
            if check_design_is_valid(x_i[0].cpu()):  # If a valid env is found, return True
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
        
        changes = np.sum(decode_grid_design(best_x_i.cpu().squeeze(),return_map= True)!=decode_grid_design(x.cpu().squeeze(),return_map= True))
        # Calculate the time taken in milliseconds
        time_taken_ms = (end_time - start_time)
        cumulatiev_time += time_taken_ms
        
        
        if i%10==1: # displaying progress
            
            true_wcds.append(compute_true_wcd(x_i[0].cpu()))
            if true_wcds[-1] is None:
                print("IS VALID ?", check_design_is_valid(x_i[0].cpu()),model(x_i).item())
                # plot_grid(decode_grid_design(x_i.cpu().squeeze(),return_map= True).tolist())
            wcds.append(wcd)
            iters.append(i)
            x_envs.append(x_i)
            print("i = ",i,"Pred:",wcd.item()," True",true_wcds[-1],"time taken",time_taken_ms)
            times.append(cumulatiev_time)
            wcd_changes.append(true_wcds[0]-true_wcds[-1])
        
        if updated_pos == -1:  
            break
        if no_progress >10:
            break
        
        true_wcds.append(compute_true_wcd(best_x_i[0].cpu()))
        wcd_changes.append(true_wcds[0]-true_wcds[-1])
        
    
    true_wcds.append(compute_true_wcd(best_x_i[0].cpu()))
    if true_wcds[-1] is None:
        print("IS VALID ?", check_design_is_valid(best_x_i[0].cpu()),model(best_x_i).item())
        # plot_grid(decode_grid_design(x_i.cpu().squeeze(),return_map= True).tolist())
    wcds.append(wcd)
    iters.append(i)
    x_envs.append(x_i)
    
    print("i = ",i,"Pred:",wcd.item()," True",true_wcds[-1],"time taken",time_taken_ms)
    times.append(cumulatiev_time)
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
        type=float,  # Accepts a float input
        default=0,    # Defaults to 0
        help="Cost parameter for the simulation. Default is 0.",
    )

    parser.add_argument(
        "--grid_size",
        type=int,  # Accepts an integer input
        default=10,  # Defaults to 10
        help="Maximum grid size.",
    )

    parser.add_argument(
        "--max_iter",
        type=int,  # Accepts an integer input
        default=20,  # Defaults to 20
        help="Maximum number of iterations.",
    )

    parser.add_argument(
        "--num_instances",
        type=int,  # Accepts an integer input
        default=500,  # Defaults to 500
        help="Spacing in the test dataset to use for the experiment",
    )
    parser.add_argument(
        "--timeout_seconds",
        type=int,  # Ensure that the input is expected to be a int
        default=600,  # Set the default value to 600
        help="Timeout seconds",
    )

    parser.add_argument(
        "--experiment",
        type=str,  # Accepts a string input
        default="ALL_MODS",  # Defaults to 'ALL_MODS'
        choices=["BLOCKING_ONLY", "BOTH_UNIFORM"],  # Restricted choices
        help="Specify either BLOCKING_ONLY or BOTH_UNIFORM for all modifications.",
    )

    args = parser.parse_args()
    cost = args.cost
    grid_size = args.grid_size
    experiment = args.experiment
    num_instances = args.num_instances
    timeout_seconds = args.timeout_seconds
    
    current_directory = os.path.dirname(os.path.realpath(__file__))
    
    dataset_label = f"{current_directory}/data/grid{grid_size}/model_training/dataset_{grid_size}_best.pkl" # DATA 
    model_label = f"{current_directory}/models/wcd_nn_model_{grid_size}_best.pt"

    with open(dataset_label, "rb") as f:
        loaded_dataset = pickle.load(f)
        
    experiment_label=experiment+ "_test"


    model = torch.load(model_label)
    model = model.to(DEVICE).eval()
    # model = model.cuda().eval()

    true_wcds_per_cost=[]
    wcds_per_cost = []
    gammas = []
    times = []
    costs = []
    num_changes = []
    max_budgets = []
    
    times = []
    all_wcd_changes = []
    realized_budgets =[]
    
    
    if experiment == "BLOCKING_ONLY":
        lambda_1_values = [0.001,0.005,0.01,0.05,0.1,0.2,0.5,1.0,5]
        lambda_2_values = [0.1,0.5,1,10,100]
        blocking_rat = 1
        unblocking_rat = 0
    else:
        lambda_1_values = [0,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1.0,2,5]
        lambda_2_values = [0]
        blocking_rat = 1.05
        unblocking_rat = 1.0
    
    data_storage_path =f"{current_directory}/data/grid{grid_size}/timeout_{timeout_seconds}"
    create_folder(data_storage_path) # create this if not exists 
    
    max_budgets = [1,3,5,7,9,11,13,15,17,19,21,23]
    create_or_update_list_file(f"{data_storage_path}/max_budgets_{grid_size}_{experiment_label}.csv",[max_budgets])
    
    for j in range(0, len(loaded_dataset),len(loaded_dataset)//num_instances):
                
            env_wcd_changes = []
            env_budgets = []
            env_times = []
            env_max_budget =[]
            max_budget = 25
            
            budget_buckets_realized = [[]]*len(max_budgets)
            budget_buckets_wcd_change = [-100]*len(max_budgets)
            budget_buckets_times = [0]*len(max_budgets)
            
            for lambda_1 in  lambda_1_values: 
                for lambda_2 in lambda_2_values:
                    print("Environment; ",j, "Langrange Multipliers: ",[lambda_1, lambda_2])
                    x, y = loaded_dataset[j]  # Get a specific data sample

                    x = x.unsqueeze(0).float().to(DEVICE)
                    # print("Device",x.unsqueeze(0).float().cuda().device,x.unsqueeze(0).float().to(DEVICE).device)
                    print("Original Predicted WCD = ",model(x).item())

                    best_wcd = []
                    invalid_envs_collection =[]
                    true_wcds = []
                    max_changes_dist=np.round([(blocking_rat*max_budget)/(unblocking_rat+blocking_rat),
                                               (unblocking_rat*max_budget)/(unblocking_rat+blocking_rat) ]).tolist()
                    max_iter = max_budget
                    best_x_i, invalid_envs,wcds,true_wcds,x_envs,iters, time_taken, wcd_changes = minimize_wcd(model, x, 
                                                                                                               lambdas=[lambda_1, lambda_2],
                                                        max_changes_dist=max_changes_dist if experiment != "BOTH_UNIFORM" else [-1,-1] ,
                                                        grid_size =grid_size, max_iter = max_iter, 
                                                                                                blocking_only = False
                                                                                                )
                    env_times.append(time_taken)
                    env_wcd_changes.append(wcd_changes[-1])
                    wcds_per_cost.append(wcds)
                    true_wcds_per_cost.append(true_wcds)

                    if len(x_envs) ==1:
                            continue

                    costs.append([lambda_1, lambda_2])
                    n_changes = np.sum(decode_grid_design(best_x_i.cpu().squeeze(),return_map= True)!=decode_grid_design(x.cpu().squeeze(),return_map= True))

                    x_changes = best_x_i.cpu()[:, 1, :, :]-x.cpu()[:, 1, :, :]
                    blockings = (x_changes==1).sum(axis=(1, 2))
                    removals = (x_changes==-1).sum(axis=(1, 2))
                    num_changes.append([blockings.item(),removals.item()])
                    wcd_change = wcd_changes[-1]
                    
                    print("Final WCD = ",model(best_x_i).item(),"True",true_wcds[-1],"n_changes",n_changes,"Bugdet",max_budget, "Time taken",time_taken)
                    
                    for i,budget in enumerate(max_budgets):
                        
                        max_changes_dist=np.round([(blocking_rat*budget)/(unblocking_rat+blocking_rat),
                                                   (unblocking_rat*budget)/(unblocking_rat+blocking_rat) ]).tolist()
                        
                        if ((blockings.item()<= max_changes_dist[0] and removals.item()<= max_changes_dist[1] ) 
                            and experiment != "BOTH_UNIFORM") or (experiment == "BOTH_UNIFORM"  and 
                                                                                       np.sum([blockings.item(),removals.item()])<=np.sum(max_changes_dist)) :
                            print([blockings.item(),removals.item()], max_changes_dist,blockings.item()<= max_changes_dist[0],removals.item()<= max_changes_dist[1])
                            
                            if wcd_change>budget_buckets_wcd_change[i]: # found a better value
                                budget_buckets_wcd_change[i] = wcd_change
                                budget_buckets_realized[i] = [blockings.item(),removals.item()]
                                budget_buckets_times[i] = time_taken
                                
                                # store this instance
                                update_or_create_dataset(f"{data_storage_path}/initial_envs_{grid_size}_{experiment}.pkl", [x_envs[0]], [true_wcds[0]]) # store the initial environments
                                update_or_create_dataset(f"{data_storage_path}/final_envs_{grid_size}_{experiment}.pkl", [x_envs[-1]], [true_wcds[-1]]) # store the final environments
                
                            
            
            # pdb.set_trace()
            times.append(budget_buckets_times)
            all_wcd_changes.append(budget_buckets_wcd_change)
            realized_budgets.append(budget_buckets_realized)
            
            
            create_or_update_list_file(f"{data_storage_path}/times_{grid_size}_{experiment_label}.csv",times)
            create_or_update_list_file(f"{data_storage_path}/wcd_change_{grid_size}_{experiment_label}.csv",all_wcd_changes)
            create_or_update_list_file(f"{data_storage_path}/budgets_{grid_size}_{experiment_label}.csv",realized_budgets)

     