
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import torch
import pandas as pd
import itertools
import math
from mdp import GridworldMDP, construct_grid, plot_grid

from collections import deque

import random
from torch.utils.data import Dataset

import pickle

import os
import re
import torch.nn as nn
import csv

from torchvision.models import resnet50, resnet18,resnet34,resnet101,vgg16

RANDOM_SEED = 42

def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(RANDOM_SEED)

class CNN4(nn.Module):
    def __init__(self, n_channels=13,drop_out=0.01):
        super(CNN4, self).__init__()

        # Replace the first seven convolutional layers with ResNet50
        self.resnet50 = resnet18(pretrained=False)

        self.resnet50.conv1=self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1)
        self.resnet50.maxpool = nn.Identity()
        # self.resnet50.bn1 = nn.BatchNorm2d(8)

        # Keep the rest of the architecture as-is
        self.fc1 = nn.Linear(1000, 32)
        self.relu8 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(0.0001)

        self.fc2 = nn.Linear(32, 16)
        self.relu9 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(drop_out)

        self.fc3 = nn.Linear(16, 1)
        self.softplus = nn.functional.softplus
        self.reluN =nn.ReLU()

    def forward(self, x):
        x = self.resnet50(x)
        # pdb.set_trace()
        # print(x.shape)
        x = x.view(-1, 1000)
        
        x = self.fc1(x)
        x = self.relu8(x)


        x = self.fc2(x)
        x = self.relu9(x)
        x = self.dropout2(x)

        x = self.reluN(self.fc3(x))
        # x = self.softplus(self.fc4(x))

        return x
    
    

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        # print(self.X.shape)
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.Y[idx], dtype=torch.float32)
        return x, y
    
    def update_data(self, new_X, new_Y):
        self.X = torch.cat([self.X, new_X])
        self.Y = torch.cat([self.Y, new_Y])
        
def update_or_create_dataset(file_name, new_X, new_Y):
    new_X = torch.cat(new_X).cpu()
    new_Y = torch.tensor(new_Y).cpu()
    file_name =f"{file_name}"
    if os.path.exists(file_name):
        # If the file exists, load the existing dataset
        with open(file_name, 'rb') as file:
            dataset = pickle.load(file)
        
        # Update the existing dataset with new data
        dataset.update_data(new_X, new_Y)
    else:
        # print("Saved : ",new_X.shape)
        # If the file doesn't exist, create a new dataset
        dataset = CustomDataset(new_X, new_Y)
    
    # # Save the updated or new dataset to the file
    with open(file_name, 'wb') as file:
        pickle.dump(dataset, file)
    
    return dataset    




def bfs_shortest_path(grid_size, start, goal, blocked):
    """
    Perform BFS to find the shortest path length from start to goal.
    """
    queue = deque([(start, 0)])  # (position, distance)
    visited = set([start])

    while queue:
        (x, y), dist = queue.popleft()

        if (x, y) == goal:
            return dist

        for neighbor in get_neighbors(grid_size,(x,y)):
            if neighbor not in visited and neighbor not in blocked:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
                
        # for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        #     nx, ny = x + dx, y + dy
        #     if 0 <= nx < n and 0 <= ny < n and (nx, ny) not in blocked and (nx, ny) not in visited:
        #         visited.add((nx, ny))
        #         queue.append(((nx, ny), dist + 1))

    return -1  # No path found

def find_shortest_paths(n,start, target, k, blocked):
    queue = deque([(start, [])])
    visited = set()
    paths = []

    while queue and k >= 0:
        curr, path = queue.popleft()
        visited.add(curr)
        if curr == target and len(path) <= k:
            paths.append(path + [curr])
            continue

        if len(path) > k:
            continue

        for neighbor in get_neighbors(n,curr):
            if neighbor not in visited and neighbor not in blocked:
                queue.append((neighbor, path + [curr]))
    return paths


                
def get_neighbors(n,curr):
    # Return a list of neighboring cells
    neighbors =[]
    x,y = curr
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n:
                neighbors.append((nx, ny))
    return neighbors


def compute_wcd_from_paths(paths1, paths2, return_wcd_paths = False):
    max_prefix_length = 0
    wcd_paths = []
    for path1 in paths1:
        for path2 in paths2:
            prefix_length = -1
            i = 0
            while i < len(path1) and i < len(path2) and path1[i] == path2[i]:
                prefix_length += 1
                i += 1
            if prefix_length > max_prefix_length:
                max_prefix_length = prefix_length
                wcd_paths = [[path1],[path2]]
                
    if return_wcd_paths:
        return max_prefix_length, wcd_paths
    else:
        return max_prefix_length


def randomize_pos(grid_size, num_goal_pos, num_special_pos, num_blocked_pos):
    # Check if the total number of positions exceeds the grid capacity
    if num_goal_pos + num_special_pos + num_blocked_pos >= grid_size**2:
        raise ValueError("Total number of positions exceeds the capacity of the grid.")

    all_possible_pos = [(i, j) for i in range(grid_size) for j in range(grid_size)]

    # Define possible positions for goal positions in the last columns
    last_column_pos = [(i, j) for i in range(grid_size) for j in range(grid_size - 2, grid_size)]
    new_goal_pos = random.sample(last_column_pos, num_goal_pos)
    
    remaining_pos = list(set(all_possible_pos) - set(new_goal_pos))

    # Define possible positions for special reward positions in the first three columns
    first_three_column_pos = list(set([(i, j) for i in range(grid_size) for j in range(2,6)]) & set(remaining_pos))
    new_special_reward_pos = random.sample(first_three_column_pos, num_special_pos)
    remaining_pos = list(set(remaining_pos) - set(new_special_reward_pos))
    
    # Ensure there are enough remaining positions for blocked positions
    if len(remaining_pos) < num_blocked_pos + 1: # +1 to account for the start position
        raise ValueError("Not enough remaining positions for blocked positions after allocating goal and special reward positions.")

    # Randomly select unique positions for blocked positions
    new_blocked_pos = random.sample(remaining_pos, num_blocked_pos)
    remaining_pos = list(set(remaining_pos) - set(new_blocked_pos))

    # Choose start position from the first column of the remaining positions
    first_column_pos = [(i, 0) for i in range(grid_size) if (i, 0) in remaining_pos]
    if not first_column_pos:
        raise ValueError("No available positions in the first column for the start position.")
    start_pos = random.choice(first_column_pos)

    return new_goal_pos, new_special_reward_pos, new_blocked_pos, start_pos


def encode_mdp_design(grid_size, goal_positions, goal_rewards, blocked_positions, start_pos, sub_goals_pos, sub_goal_rewards, gamma):
    # Creating a 3D array with 4 channels, each of size grid_size x grid_size
    channels = np.zeros((5, grid_size, grid_size))  # start_pos, blocked_channel, subgoal_positions+rewards channel, goal_position+value channel

    # Marking the blocked spaces
    for blocked in blocked_positions:
        channels[1, blocked[0], blocked[1]] = 1

    # Marking the starting position
    channels[0, start_pos[0], start_pos[1]] = 1

    # Marking the subgoal positions with their corresponding values
    for sub_goal, sub_goal_value in zip(sub_goals_pos, sub_goal_rewards):
        channels[2, sub_goal[0], sub_goal[1]] = sub_goal_value
        
    # Marking the goal positions with their corresponding values
    for goal, goal_value in zip(goal_positions, goal_rewards):
        channels[3, goal[0], goal[1]] = goal_value
        
    

    # Filling the entire gamma channel with the gamma value
    channels[4, :, :] = gamma

    return torch.tensor(channels).unsqueeze(0)

def decode_mdp_design(encoded_tensor, return_grid = False, K = None):
    # Ensure that the input is a PyTorch tensor
    if not isinstance(encoded_tensor, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor")

    # Convert the tensor to a NumPy array
    channels = encoded_tensor.squeeze(0).numpy()

    # Extracting information from channels
    blocked_positions = np.argwhere(channels[1] == 1)
    blocked_positions = [tuple(pos) for pos in blocked_positions]
    start_pos = tuple(np.argwhere(channels[0] == 1)[0])
    
    sub_goal_positions = []
    sub_goal_rewards = []

    for i, row in enumerate(channels[2]):
        for j, value in enumerate(row):
            if value != 0:
                sub_goal_positions.append((i, j))
                sub_goal_rewards.append(value)
                
    goal_positions = []
    goal_rewards = []

    for i, row in enumerate(channels[3]):
        for j, value in enumerate(row):
            if value != 0:
                goal_positions.append((i, j))
                goal_rewards.append(value)
    
    if K:
        gamma = K
    else:
        print("K is ",K)
        gamma = channels[4, 0, 0]  # Assuming gamma is the same for all positions
        print("K is ",gamma)
    grid_size =channels[3].shape[0] 
    # Return the decoded information
    
    decoded_info =  {
        "grid_size":grid_size,
        "blocked_positions": blocked_positions,
        "start_pos": start_pos,
        "sub_goal_positions": sub_goal_positions,
        "sub_goal_rewards": sub_goal_rewards,
        "goal_positions": goal_positions,
        "goal_rewards": goal_rewards,
        "gamma": gamma
    }
    
    if return_grid:
        return decoded_info,construct_grid(decoded_info)
    else:  
        return decoded_info


def extract_positions(grid):
    start_pos = None
    blocked_positions = []
    goal_positions = []
    sub_goal_positions = []
    space_positions = []
    goal_rewards = []
    sub_goal_rewards = []
    grid_size = len(grid)
    for x, row in enumerate(grid):
        for y, cell in enumerate(row):
            if cell == 'S':
                start_pos = (x, y)
            elif cell == 'X':
                blocked_positions.append((x, y))
            elif cell == ' ':
                continue
            else: # extract the values
                val = float(cell)
                if val < 1: # subgoal
                    sub_goal_positions.append((x,y))
                    sub_goal_rewards.append(val)
                else: # its a goal
                    goal_positions.append((x,y))
                    goal_rewards.append(val)
    
    decoded_info =  {
        "grid_size":grid_size,
        "blocked_positions": blocked_positions,
        "start_pos": start_pos,
        "sub_goal_positions": sub_goal_positions,
        "sub_goal_rewards": sub_goal_rewards,
        "goal_positions": goal_positions,
        "goal_rewards": goal_rewards,
        "gamma": 0.0
    }
    
    return decoded_info

def encode_from_grid_to_x(grid, gamma):
    encoding = extract_positions(grid)
    encoding["gamma"] = gamma
    
    grid_size = encoding["grid_size"]
    blocked_positions = encoding["blocked_positions"]
    start_pos = encoding["start_pos"]
    sub_goals_pos = encoding["sub_goal_positions"]
    sub_goal_rewards = encoding["sub_goal_rewards"]
    goal_positions  = encoding["goal_positions"]
    goal_rewards = encoding["goal_rewards"]
    gamma= encoding["gamma"]
    
    return encode_mdp_design(grid_size, goal_positions, goal_rewards, blocked_positions, start_pos, sub_goals_pos, sub_goal_rewards, gamma)

def is_x_valid(x, K = None):
    
    encoding = decode_mdp_design(x,K=K)
    # print(encoding)
    grid_size = encoding["grid_size"]
    blocked_states = encoding["blocked_positions"]
    start_pos = encoding["start_pos"]
    special_reward_states = encoding["sub_goal_positions"]
    special_rewards = encoding["sub_goal_rewards"]
    goal_states  = encoding["goal_positions"]
    goal_state_rewards = encoding["goal_rewards"]
    
    if K:
        gamma = K
    else:
        gamma= encoding["gamma"]
    
    return is_design_valid(grid_size, goal_states, blocked_states, start_pos)
    
    

def is_design_valid(grid_size, goal_positions, blocked_positions, start_pos):
    shortest_path_lens =[]
    for i,goal_pos in enumerate(goal_positions):
        temp_blocked_positions = blocked_positions+[goal_positions[1-i]]
        shortest_path_len = bfs_shortest_path(grid_size, start=start_pos, goal=goal_pos, blocked=temp_blocked_positions)
        shortest_path_lens.append(shortest_path_len)
            # return False  # Invalid environment, as there's no path to a goal
    return not (-1 in shortest_path_lens), shortest_path_lens

# def decode_grid_design(encoded_grid, return_map = False):
#     # Ensure that the input is a numpy array for easy handling
#     if isinstance(encoded_grid, torch.Tensor):
#         encoded_grid = encoded_grid.numpy()
    
#     # Extract the channels
#     spaces, blocked, start, goals = encoded_grid

#     # Initialize an empty grid for visualization
#     n = spaces.shape[1]  # Assuming square grid
#     grid = np.full((n, n), ' ')
   

#     # Mark start and goal positions
#     start_pos = np.argwhere(start == 1) # the list
#     goal_pos = np.argwhere(goals == 1)
#     blocked_pos = np.argwhere(blocked == 1)
#     spaces_pos = np.argwhere(spaces == 1)
    
#     start_position = tuple(start_pos.tolist()[0]) # the single point
    
#     if not return_map: # extract the positions 
#         goal_pos =  [ tuple(pos) for pos in goal_pos.tolist()]
#         blocked_pos = [tuple(pos) for pos in blocked_pos.tolist() if tuple(pos) != start_position]
#         spaces_pos = [tuple(pos) for pos in spaces_pos.tolist() if tuple(pos) != start_position]
#         return n, goal_pos, blocked_pos ,start_position ,spaces_pos
    
#      # Mark blocked positions
#     for i in range(n):
#         for j in range(n):
#             if blocked[i, j] == 1:
#                 grid[i, j] = 'X'
#     if len(start_pos) > 0:
#         grid[start_pos[0, 0], start_pos[0, 1]] = 'S'
    
#     for i,pos in enumerate(goal_pos):
#         grid[pos[0], pos[1]] = f'G{i}'
    
#     return grid


def compute_wcd_single_env(grid_size, start_pos, blocked_states, goal_states, goal_state_rewards, special_reward_states, special_rewards, gamma, vis_paths=False, return_paths=False):
    """
    Compute Worst Case Distinctiveness for a single environment.

    Parameters:
    - grid_size (int): Size of the gridworld.
    - start_pos (tuple): Starting position.
    - blocked_states (list): List of new blocked states.
    - goal_states (list): List of new goal states.
    - goal_state_rewards (list): List of rewards for each goal state.
    - special_reward_states (list): List of positions with special rewards.
    - special_rewards (list): List of special rewards.
    - gamma (float): Discount factor.
    - vis_paths (bool): Whether to visualize paths (default is False).
    - return_paths (bool): Whether to return paths (default is False).

    Returns:
    - wcd (float): Worst Case Distinctiveness.
    """
    
    if not is_design_valid(grid_size, goal_states, blocked_states, start_pos)[0]:  # invalid env
        return None

    paths = []
    for i in range(len(goal_states)):
        new_blocked_states = blocked_states + [goal_states[i - 1]]
        
        mdp = GridworldMDP(n=grid_size, goal_state_pos=[goal_states[i]], goal_state_rewards=[goal_state_rewards[i]], 
                           blocked_pos=new_blocked_states, special_reward_pos=special_reward_states, 
                           special_rewards=special_rewards, start_pos=start_pos)
        path1 = simulate_agent_hyperbolic(mdp, gamma, visualize=vis_paths)

        paths.append(path1)

    wcd = compute_wcd_from_paths([paths[0]], [paths[1]])

    if return_paths:
        return wcd, paths
    else:
        return wcd

def simulate_agent_hyperbolic(mdp, gamma=1.0, visualize = True):
    policies = compute_hyperbolic_policy(mdp, gammas = [gamma])
    paths = []
    
    policy = policies[0]
    # print("Gamma: ",gamma)
    path = [mdp.reset()[0]]
    for t in range(0,20):
        idx = mdp.get_curr_state_index()
        next_pos,_,reward = mdp.move(mdp.index_action(policy[idx]))
        path.append(next_pos)
    if visualize:
        mdp.visualize(path)
    
    return path

def compute_hyperbolic_policy(mdp, gammas = [0.1, 0.2]):
    policies = []
    for gamma in gammas:
        policies.append(mdp.computeHyperbolicQ(gamma=gamma, T=20))
    return policies

def compute_true_wcd(x, vis_paths = False, K = None):
    
    encoding = decode_mdp_design(x,K=K)
    # print(encoding)
    grid_size = encoding["grid_size"]
    blocked_states = encoding["blocked_positions"]
    start_pos = encoding["start_pos"]
    special_reward_states = encoding["sub_goal_positions"]
    special_rewards = encoding["sub_goal_rewards"]
    goal_states  = encoding["goal_positions"]
    goal_state_rewards = encoding["goal_rewards"]
    gamma= encoding["gamma"]
    
    
    return compute_wcd_single_env(grid_size=grid_size, start_pos=start_pos, blocked_states=blocked_states,
                                  goal_states=goal_states, goal_state_rewards =goal_state_rewards, 
                                  special_reward_states = special_reward_states, 
                                  special_rewards=special_rewards, gamma=gamma, vis_paths=vis_paths, return_paths=False)



    
def check_design_is_valid(x, is_grid = False, K=None):
    if is_grid:
        
        encoding = extract_positions(x)

        grid_size = encoding["grid_size"]
        blocked_positions = encoding["blocked_positions"]
        start_pos = encoding["start_pos"]
        sub_goals_pos = encoding["sub_goal_positions"]
        sub_goal_rewards = encoding["sub_goal_rewards"]
        goal_positions  = encoding["goal_positions"]
        goal_rewards = encoding["goal_rewards"]
        gamma= encoding["gamma"]
        
        return  is_design_valid(grid_size, goal_positions, blocked_positions, start_pos)
    
    else:
        return is_x_valid(x,K)


def create_or_update_list_file(filename, number_list):
    """
    Appends a list of numbers to a CSV file. Creates a new file if it doesn't exist.

    Args:
    filename (str): The name of the CSV file.
    number_list (list): A list of numbers to append to the file.
    """
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(number_list)