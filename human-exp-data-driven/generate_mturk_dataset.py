"""Converted from Generate Mturk Dataset.ipynb.
Generated automatically by tools/notebook_to_script.py.
"""

import numpy as np
import pickle
import sys
import torch
from torch.utils.data import Dataset
from utils import *
import random
import matplotlib.pyplot as plt
from torchvision.models import resnet50, resnet18
import argparse
import traceback
import seaborn as sns
import pandas as pd
import csv
import ast
import pdb
from optmize_wcd_human_exp import minimize_wcd
from utils_human_exp import compute_humal_model_wcd

# %% [code] cell 0
# sys.path.insert(0, "./")
# sys.path.insert(0, "../../")

# %% [code] cell 1
GRID_SIZE =6
grid_size = GRID_SIZE
time_out = 600
assumed_behavior = "HUMAN"
interval = 335

# %% [code] cell 2
env_ids = np.array([101, 108,  98,  13, 112,  29,  95,  77, 120,  88,  52, 126,  81,
       137, 100, 115, 136,  72,  14,   6, 130, 104,  76,  78,  83,  62,
        99, 116, 119,  38])
human_model = torch.load(f'models/human_model_grid{grid_size}.pt', map_location=torch.device('cpu'))
def run_optimization(assumed_behavior="OPTIMAL"):
    human_best_lambdas = np.array(pd.read_csv('./data/grid6/HUMAN/HUMAN_BOTH_UNIFORM_test/n_lambdas_17/best_lambdas_6_HUMAN_BOTH_UNIFORM_test.csv',header=None))
    optimal_best_lambdas = np.array(pd.read_csv('./data/grid6/OPTIMAL/OPTIMAL_BOTH_UNIFORM_test/n_lambdas_17/best_lambdas_6_OPTIMAL_BOTH_UNIFORM_test.csv',header = None))
    human_best_lambdas = human_best_lambdas[env_ids]
    optimal_best_lambdas = optimal_best_lambdas[env_ids]

    dataset_label = f"data/grid{grid_size}/model_training/dataset_{grid_size}_best.pkl"
    if assumed_behavior=="OPTIMAL":
        model_label = f"models/wcd_nn_model_{grid_size}_best_optimal.pt"
        best_lambdas = optimal_best_lambdas
    else:
        model_label = f"models/wcd_nn_model_{grid_size}_best_human.pt"
        best_lambdas = human_best_lambdas

    with open(dataset_label, "rb") as f:
        loaded_dataset = pickle.load(f)


    device ="cuda:0"
    model = torch.load(model_label)
    model = model.to(device).eval()

    for i,k in enumerate(env_ids):
        x, y = loaded_dataset[k*interval]  # Get a specific data sample
        x = x.unsqueeze(0).float().cuda()
        lambdas = eval(best_lambdas[i][-1])
        print(k,k*interval, lambdas)
        # print(human_best_lambdas[i])
        best_x_i, invalid_envs,wcds,true_wcds,x_envs,iters, time_taken, wcd_changes = minimize_wcd(model,
                                                                                                                       x,
                                                                                                    lambdas=lambdas,
                                                                max_changes_dist=[1,1] ,
                                                                grid_size =grid_size, max_iter = 25
                                                                                                        )
        print("Final WCD",true_wcds[-1])
        torch.save(x_envs[-1],f"./data/grid6/{assumed_behavior}/final_envs/env_{k*interval}.pt")

# %% [code] cell 3
run_optimization(assumed_behavior="OPTIMAL")
run_optimization(assumed_behavior="HUMAN")

# %% [code] cell 4
48*335

# %% [code] cell 5
original_envs = []
greedy_true_envs =[]
optimal_model_envs = []
human_model_envs =[]

dataset_label = f"data/grid{grid_size}/model_training/dataset_{grid_size}_best.pkl"
with open(dataset_label, "rb") as f:
    loaded_dataset = pickle.load(f)

for env_id in env_ids:
    x,y = loaded_dataset[env_id*interval]
    original_envs.append(x.unsqueeze(0))

    x_o=torch.load(f"./data/grid6/OPTIMAL/final_envs/env_{env_id*interval}.pt")
    optimal_model_envs.append(x_o)

    x_h=torch.load(f"./data/grid6/HUMAN/final_envs/env_{env_id*interval}.pt")
    human_model_envs.append(x_h)

    x_g=torch.load(f"./baselines/data/grid6/timeout_600/HUMAN/BOTH_UNIFORM_GREEDY_TRUE_WCD/final_envs/env_{env_id*interval}_budget19.pt")
    greedy_true_envs.append(x_g)

# %% [code] cell 7
original_envs = torch.cat(original_envs)
greedy_true_envs = torch.cat(greedy_true_envs)
optimal_model_envs = torch.cat(optimal_model_envs)
human_model_envs = torch.cat(human_model_envs)

# %% [code] cell 8
# original_envs.shape

# %% [code] cell 9
original_human_wcds = []
for x_i in original_envs:
    original_human_wcds.append(compute_humal_model_wcd(x_i.cpu(),model =human_model))

# %% [code] cell 10
optimal_model_human_wcds = []
for x_i in optimal_model_envs:
    optimal_model_human_wcds.append(compute_humal_model_wcd(x_i.cpu(),model =human_model))

# %% [code] cell 11
data_model_human_wcds = []
for x_i in human_model_envs:
    data_model_human_wcds.append(compute_humal_model_wcd(x_i.cpu(),model =human_model))

# %% [code] cell 12
greedy_human_wcds = []
for x_i in greedy_true_envs:
    greedy_human_wcds.append(compute_humal_model_wcd(x_i.cpu(),model =human_model))

# %% [code] cell 13
optimal_wcd_change = np.array(pd.read_csv('./data/grid6/OPTIMAL/OPTIMAL_BOTH_UNIFORM_test/n_lambdas_17/wcd_change_6_OPTIMAL_BOTH_UNIFORM_test.csv',header=None))
human_wcd_change = np.array(pd.read_csv('./data/grid6/HUMAN/HUMAN_BOTH_UNIFORM_test/n_lambdas_17/wcd_change_6_HUMAN_BOTH_UNIFORM_test.csv',header=None))
greedy_wcd_change = np.array(pd.read_csv(f'./baselines/data/grid6/timeout_600/HUMAN/BOTH_UNIFORM_GREEDY_TRUE_WCD/wcd_change_6_BOTH_UNIFORM_GREEDY_TRUE_WCD.csv', header=None))

greedy_wcd_change = greedy_wcd_change[env_ids][:,-1]
optimal_wcd_change = optimal_wcd_change[env_ids][:,-1]
human_wcd_change = human_wcd_change[env_ids][:,-1]

# %% [code] cell 14
human_wcd_change>=greedy_wcd_change

# %% [code] cell 15
human_wcd_change == (np.array(original_human_wcds)-np.array(data_model_human_wcds))

# %% [code] cell 17
optimal_wcd_change == (np.array(original_human_wcds)-np.array(optimal_model_human_wcds))

# %% [code] cell 18
greedy_wcd_change == (np.array(original_human_wcds)-np.array(greedy_human_wcds))

# %% [code] cell 19
torch.save(human_model_envs,"./data/grid6/mturk_validation/our_approach_data_driven.pt")
torch.save(optimal_model_envs,"./data/grid6/mturk_validation/our_approach_optimal_model.pt")
torch.save(greedy_true_envs,"./data/grid6/mturk_validation/greed_true_data_driven.pt")
torch.save(original_envs,"./data/grid6/mturk_validation/original_environments.pt")
