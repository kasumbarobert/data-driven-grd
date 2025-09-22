import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import torch
import pandas as pd
import itertools
import math


from collections import deque
from utils_human_exp import *
import random
import argparse

human_model = torch.load('model_grid6.pt', map_location=torch.device('cpu'))
human_model.eval()
    
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
        
        wcd = compute_human_wcd(grid_size, goal_positions = new_goal_positions, blocked_positions = new_blocked_positions, start_pos = start_pos, model = human_model)
        if wcd:
            wcd_results.append(wcd)
            designs.append(encode_grid_design(grid_size, new_goal_positions, new_blocked_positions, start_pos))

    return designs,wcd_results
    

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Continuous Data Generation')

    # Add command-line argument for grid_size
    parser.add_argument('--grid_size', type=int, default=10, help='Size of the grid')
    # Add command-line argument for grid_size
    for i in range(0, 20000):
    # try:
    # Example usage
        grid_size = 6
        num_goal_positions = 2
        max_iterations = 50  # Set the number of iterations

        x,y = continuous_design_wcd(grid_size, num_goal_positions, max_iterations)
        print(y)

        file_name = f"data/grid6/model_training/tmp_human_path.pt"
        update_or_create_dataset(file_name, x, y)

if __name__ == "__main__":
    main()