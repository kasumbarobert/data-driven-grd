"""Converted from Grid Design Generation.ipynb.
Generated automatically by tools/notebook_to_script.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import torch
import pandas as pd
import itertools
import math
from mdp import GridworldMDP
from collections import deque
from utils import *

# %% [markdown]
# # use this to generate new environment designs with the corresponding dataset

# %% [code] cell 2
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

    for _ in range(max_iterations):
        num_blocked_positions = random.choice(range(grid_size*2))
        new_goal_positions, new_special_reward_positions, new_blocked_positions, start_pos = randomize_pos(grid_size, num_goal_positions, 0, num_blocked_positions)

        wcd = compute_wcd_single_env_no_paths(grid_size, new_goal_positions, new_blocked_positions, start_pos)
        if wcd:
            wcd_results.append(wcd)
            designs.append(encode_grid_design(grid_size, new_goal_positions, new_blocked_positions, start_pos))

    return designs,wcd_results

# %% [code] cell 3
for i in range(0, 500):
    # try:
        # Example usage
    grid_size = 13
    num_goal_positions = 2
    max_iterations = 100  # Set the number of iterations

    x,y = continuous_design_wcd(grid_size, num_goal_positions, max_iterations)

    file_name = f"./data/grid{grid_size}/model_training/dataset_wcd.pt"
    update_or_create_dataset(file_name, x, y)
    # except:
    #     pass
