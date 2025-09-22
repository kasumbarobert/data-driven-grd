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
from utils_suboptimal import *
import random
import argparse
import time


    
def continous_data_generation(grid_size = 7,num_goal_states = 2, K = 4, file_name = f"_"):
    x = []
    y = []
    for _ in range(100):
        # Define the parameters
        num_special_states =random.choice(range(1,5))
        num_blocked_states = random.choice(range(grid_size*2))

        goal_state_rewards1 = [round(random.uniform(3, 4), 1) for _ in range(num_goal_states)]
        # Generate new states
        new_goal_states, new_special_reward_states, new_blocked_states, start_pos = randomize_pos(grid_size, num_goal_states, num_special_states, num_blocked_states)
        special_rewards = [round(random.uniform(0.1, 0.5), 3) for _ in range(num_special_states)]
        
        # mdp = GridworldMDP(n=grid_size,goal_state_pos = new_goal_states,goal_state_rewards=goal_state_rewards1[0:num_goal_states] ,
        #            blocked_pos=new_blocked_states, special_reward_pos=new_special_reward_states,special_rewards=special_rewards,start_pos=start_pos)
        # mdp.visualize()

        #is_design_valid(grid_size, goal_positions, blocked_positions, start_pos)
        is_valid = is_design_valid(grid_size, new_goal_states, new_blocked_states, start_pos)[0]

        if is_valid:
            paths = []
            gammas = [K]
            
            wcds = []

            for gamma in gammas:
                x_g = encode_mdp_design(grid_size, new_goal_states, goal_state_rewards1[0:num_goal_states], new_blocked_states, start_pos, 
                              new_special_reward_states, special_rewards, gamma)
                start_time = time.time()
                wcd = compute_true_wcd(x_g)
                print("Took ", time.time()-start_time," seconds")
                x.append(x_g)
                y.append(wcd)
                wcds.append(wcd)
    update_or_create_dataset(file_name,x,y)
    

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Continuous Data Generation')

    # Add command-line argument for grid_size
    parser.add_argument('--grid_size', type=int, default=10, help='Size of the grid')
    # Add command-line argument for grid_size
    parser.add_argument('--K', type=int, default=4, help='K value')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Run the code with the specified grid_size
    grid_size = args.grid_size
    for iteration in range(0, 10000):
        if iteration % 40 == 0:
            print(iteration)
        continous_data_generation(grid_size=grid_size, num_goal_states=2, K = args.K,
                                   file_name=f"./data/grid{grid_size}/model_training/hyperbol_simulated_envs_K{args.K}_0.pkl")

if __name__ == "__main__":
    main()