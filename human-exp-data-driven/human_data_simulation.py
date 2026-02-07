"""Converted from human_data_simulation.ipynb.
Generated automatically by tools/notebook_to_script.py.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import pandas as pd
import itertools
import math
from collections import deque
import argparse
from utils_human_exp import *

# %% [markdown]
# ## generate human path from human model

# %% [code] cell 1
# from mdp import GridworldMDP

# from utils_suboptimal import *

# %% [code] cell 4
## load human model
human_model = torch.load('models/human_model_grid10.pt', map_location=torch.device('cpu'))
human_model.eval()

# %% [code] cell 5
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

grid_size = 6
num_goal_positions = 2
max_envs = 10

# %% [code] cell 6
def continuous_design_wcd(grid_size, num_goal_positions, max_iterations=100):
    """
    Continuously creates designs and computes WCD.
    Ignores environments where the shortest path to any goal is -1.
    """

    def is_valid_environment(paths):
        """ Check if the environment is valid (no shortest path is -1) """
        return all(len(path) > 0 for path in paths)

    wcd_results = []
    designs = []
    count = 0
    valid_counts = 0
    for _ in range(max_iterations):
        num_blocked_positions = random.choice(range(grid_size*2))
        new_goal_positions, new_special_reward_positions, new_blocked_positions, start_pos = randomize_pos(grid_size,
                                                                                                           num_goal_positions, 0, num_blocked_positions)
        is_valid = is_design_valid(grid_size, new_goal_positions, new_blocked_positions, start_pos)
        if not is_valid:
            continue
        valid_counts+=1
        wcd = compute_human_wcd(model = human_model, grid_size=grid_size, goal_positions = new_goal_positions,
                                blocked_positions = new_blocked_positions, start_pos = start_pos,search_depth = 19)

        #compute_human_wcd(model, grid_size, goal_positions, blocked_positions, start_pos, search_depth =19)
        x= encode_grid_design(grid_size, new_goal_positions, new_blocked_positions, start_pos).squeeze()

        grid = decode_grid_design(x,return_map = True)

        print(wcd)
        # plot_grid(grid)
        if wcd is None:
            count+=1
            wcd_results.append(wcd)
            designs.append(encode_grid_design(grid_size, new_goal_positions, new_blocked_positions, start_pos))
    print("Percent failed",count/valid_counts)
    return designs,wcd_results

# %% [code] cell 7
for i in range(0, 5):
    grid_size = 10
    num_goal_positions = 2
    max_iterations = 20  # Set the number of iterations
    x,y = continuous_design_wcd(grid_size, num_goal_positions, max_iterations)
    print(y)
    file_name = f"data/grid{grid_size}/model_training/tmp_human_path.pt"
    # update_or_create_dataset(file_name, x, y)
    # except:
    #     pass
