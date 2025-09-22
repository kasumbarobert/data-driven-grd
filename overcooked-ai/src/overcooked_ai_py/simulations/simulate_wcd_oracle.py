"""
Data Generation Script for Overcooked-AI Goal Recognition Design

Author: Robert Kasumba (rkasumba@wustl.edu)

This script generates training data for the data-driven goal recognition design approach
in the Overcooked-AI domain. It creates environment designs and computes their corresponding
Worst-Case Distance (WCD) values using oracle simulation.

WHY THIS IS NEEDED:
- The optimization process requires a large dataset of environment designs with known WCD values
- WCD computation is computationally expensive, so we pre-compute these values
- Different simulation types allow for various experimental setups and analysis

HOW IT WORKS:
1. Generates random Overcooked-AI environment layouts
2. Computes true WCD values using oracle simulation (exhaustive search)
3. Saves environment designs and WCD pairs for training the CNN oracle
4. Supports three simulation modes:
   - random: Generate random layouts with varying gamma values
   - stored: Use pre-existing layout files for consistent evaluation
   - fixed_gamma: Generate data with fixed gamma values for analysis

USAGE:
    python simulate_wcd_oracle.py --simulation-type random --max_grid_size 6

OUTPUT:
    Training data files in pickle format containing (environment_design, WCD_value) pairs
"""

import sys
sys.path.insert(0, "../../")
import overcooked_ai_py.mdp.overcooked_env as Env
import overcooked_ai_py.mdp.overcooked_mdp as Mdp
from overcooked_ai_py.mdp.actions import Action, Direction
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import pandas as pd
from os.path import exists
import pickle
from wcd_simulation_utils import *
import argparse
import csv


def simulate_randomly_generated(max_grid_size):
    """Generate random Overcooked-AI environments and compute their WCD values."""
    # Load a default dataset to use as template for environment structure
    with open("data/simulated_envs_default.pkl", "rb") as f:
        loaded_dataset = pickle.load(f)
    counter = 0

    while True:  # Infinite loop to continuously generate data
        # Create filename for batch of 200 environments
        file_name = f"data/grid{max_grid_size}/model_training/simulated_envs_{counter//200}_size{max_grid_size}.pkl"
        
        # Generate random valid Overcooked-AI layouts
        envs = np.array(generate_grids(random.randint(6, max_grid_size),valid_env=True))
        # Encode environments into tensor format for processing
        env_encoded = encode_multiple_envs([envs],grid_size=max_grid_size).squeeze(0)
        
        true_wcds = []  # Store computed WCD values
        env_xs = []     # Store environment tensors
        
        for i in range(0, len(envs)):
            # Get a random template from the loaded dataset
            x,y = loaded_dataset[random.randint(0, len(loaded_dataset))]
            # Resize to target grid size
            x = x[:,0:max_grid_size,0:max_grid_size].cpu()
            
            # Set gamma value for agent behavior (0.9999 for near-optimal)
            # gamma = randomly_choose_gamma(ranges = [(0.65, 1.0)], probabilities = [1.0])
            gamma = 0.9999
            
            # Replace environment layout with generated one (channels 0-7)
            x[0:8,:,:] = torch.tensor(env_encoded[i])
            # Set gamma value across all positions (channel 8)
            x[8,:,:] = torch.full((max_grid_size, max_grid_size), gamma)
            
            # Compute true WCD using oracle simulation (exhaustive search)
            true_wcd = compute_true_wcd(x.unsqueeze(0),grid_size=max_grid_size)
            true_wcds.append(true_wcd)
            env_xs.append(x.unsqueeze(0))
            
        # Save batch of environments and their WCD values
        update_or_create_dataset(file_name, env_xs, true_wcds)
        counter += 1
        
    # Clear GPU memory to prevent memory issues
    torch.cuda.empty_cache()
    

def simulate_fixed_gamma(max_grid_size):
    with open("data/simulated_envs_default.pkl", "rb") as f:
        loaded_dataset = pickle.load(f)
    counter = 0

    while True:
        file_name = f"sim_envs_fixed_gamma_{counter//200}.pkl"
        envs = np.array(generate_grids(random.randint(6, max_grid_size),valid_env=True))
        env_encoded = encode_multiple_envs([envs]).squeeze(0)
        true_wcds = []
        env_xs = []
    
        x_i,y_i = loaded_dataset[random.randint(0, len(loaded_dataset)-1)]
        wcd_per_gamma = []
        for gamma in np.arange(0, 1.04, 0.05):
            x = x_i.clone()
            x[0:8,:,:] = torch.tensor(env_encoded[0])
            x[8,:,:] = torch.full((6,6), gamma)
            true_wcd = compute_true_wcd(x.unsqueeze(0),grid_size=6)
            true_wcds.append(true_wcd)
            wcd_per_gamma.append(true_wcd)
            env_xs.append(x.unsqueeze(0))
            
        update_or_create_dataset(file_name, env_xs, true_wcds)
        counter += 1
                
        with open('wcd_per_gamma.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(wcd_per_gamma)
                
    torch.cuda.empty_cache()

        

def simulate_using_stored_files():
    gammas = [0.999,0.95,0.9,0.85,0.80,0.7]
    agent_types = [1,2,3,4,5]
    for size in range(6,7):
        
        with open(f'../data/layouts/random_generation/envs_layouts_size{size}.pkl', 'rb') as f:
            envs = pickle.load(f)
        
        for i in range(2,len(envs)):
            layout = f"envs_layouts_size{size}_{i} "
            print("============ LAYOUT FILE", layout,".layout======")
            data_set = []
            for j in range(10): # sample 10 gammas
                gamma = random.uniform(0.7, 1.0)
                data_set.append(simulate(gamma, envs[i],label=f"size_{size}_{i}"))

            sim_data_env_df = pd.DataFrame(data_set)
            if exists("data/wcd_oracle_data.csv"):
                saved_df = pd.read_csv("data/wcd_oracle_data.csv")
                pd.concat([saved_df,sim_data_env_df]).to_csv("data/wcd_oracle_data.csv",index=False)
            else:
                sim_data_env_df.to_csv("data/wcd_oracle_data.csv",index=False)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate data")
    parser.add_argument(
        "--simulation-type",
        choices=["random", "stored","fixed_gamma"],
        default="random",
        help="Simulation type: 'random' (default) or 'stored'",
    )
    parser.add_argument(
        "--max_grid_size",
        type=int,  # Ensure that the input is expected to be a float
        default=6,  # Set the default value to 6
        help="Grid size. Default is 10.",
    )

    args = parser.parse_args()

    if args.simulation_type == "stored":
        simulate_using_stored_files()
    if args.simulation_type == "fixed_gamma":
        simulate_fixed_gamma(max_grid_size = args.max_grid_size)
    else:
        # Default to simulate_randomly_generated() if no argument is provided or "random" is specified
        simulate_randomly_generated(max_grid_size = args.max_grid_size)