"""Converted from Compute Initial WCD.ipynb.
Generated automatically by tools/notebook_to_script.py.
"""

import pickle
import sys
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

# %% [code] cell 0
# sys.path.insert(0, "./")
# sys.path.insert(0, "../../")

# %% [code] cell 1
GRID_SIZE = 6
K = 8
dataset_label = f"data/grid6/model_training/dataset_{GRID_SIZE}_K8_best.pkl"
model_label = f"models/wcd_nn_model_{GRID_SIZE}_K8_best.pt"

with open(dataset_label, "rb") as f:
    loaded_dataset = pickle.load(f)

# %% [code] cell 2
num_envs = len(loaded_dataset)
init_true_wcd ={}
for i in range(num_envs):
    x, y = loaded_dataset[i]  # Get a specific data sample
    init_true_wcd[i] = compute_true_wcd(x.cpu(),K=8)
    if (i+1)%200==0:
        print(i)
        data_storage_path =f"data/grid{GRID_SIZE}/K{K}/"
        with open(f"{data_storage_path}initial_true_wcd_by_id.json", "w") as json_file:
            json.dump(init_true_wcd, json_file, indent=4)

# %% [code] cell 3
# # print(env_dict)
# data_storage_path =f"data/grid{GRID_SIZE}/K{K}/"
# with open(f"{data_storage_path}/individual_envs/initial_true_wcd_by_id.json", "w") as json_file:
#     json.dump(init_true_wcd, json_file, indent=4)
