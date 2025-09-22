"""
General Utilities for Overcooked-AI Goal Recognition Design

Author: Robert Kasumba (rkasumba@wustl.edu)

This module provides general utility functions for data processing, environment manipulation,
file I/O operations, and visualization in the Overcooked-AI goal recognition design system.

WHY THIS IS NEEDED:
- Common data processing operations used across multiple scripts
- Environment encoding/decoding between different formats
- File I/O utilities for dataset management
- Visualization tools for environment and result analysis
- Helper functions for optimization and analysis workflows

KEY FUNCTIONALITIES:
1. **Dataset Management**: Custom dataset classes and data loading utilities
2. **Environment Processing**: Encoding/decoding between Overcooked-AI and tensor formats
3. **File Operations**: CSV, pickle, and other file format utilities
4. **Visualization**: Environment and result visualization tools
5. **Data Analysis**: Statistical and analysis helper functions
6. **Optimization Support**: Utilities for gradient-based optimization

CORE UTILITIES:
- CustomDataset: PyTorch dataset wrapper for environment data
- encode_env()/decode_env(): Environment format conversion
- create_or_update_list_file(): CSV file management
- update_or_create_dataset(): Pickle dataset management
- print_grid(): Environment visualization
- compute_changes(): Environment modification tracking

USAGE:
    from utils import CustomDataset, encode_env, decode_env, create_or_update_list_file
    dataset = CustomDataset(X, Y)
    env_tensor = encode_env(environment_dict)
"""

from torch.utils.data import Dataset

import sys
sys.path.insert(0, "../../")
import overcooked_ai_py.mdp.overcooked_env as Env
import overcooked_ai_py.mdp.overcooked_mdp as Mdp
from overcooked_ai_py.mdp.actions import Action, Direction
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import pickle
from queue import PriorityQueue
import pdb
from joblib import dump, load
import pandas as pd
import pickle
from collections import deque
from queue import Queue
from collections import Counter
import csv
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer


import random

import os
import re

class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset for Overcooked-AI environment data.
    
    This dataset class handles environment tensors (X) and corresponding WCD values (Y)
    for training the CNN oracle and other machine learning models.
    """
    def __init__(self, X, Y):
        """
        Initialize dataset with environment tensors and WCD values.
        
        Args:
            X: Environment tensors of shape (N, channels, height, width)
            Y: Corresponding WCD values of shape (N,)
        """
        self.X = X
        self.Y = Y

    def __len__(self):
        """Return the number of samples in the dataset."""
        print(self.X.shape)
        return self.X.shape[0]

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            tuple: (environment_tensor, wcd_value) as torch tensors
        """
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.Y[idx], dtype=torch.float32)
        return x, y
    
    def update_data(self, new_X, new_Y):
        """
        Add new data to the existing dataset.
        
        Args:
            new_X: New environment tensors to add
            new_Y: New WCD values to add
        """
        self.X = torch.cat([self.X, new_X])
        self.Y = torch.cat([self.Y, new_Y])
        print("Saved : ", self.X.shape)

import torch.nn as nn

from torchvision.models import resnet50, resnet18,resnet34,resnet101,vgg16

import torch
import torch.nn as nn


import timm

from torchvision.models import vgg16

import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(self, n_channels=13):
        super(CustomCNN, self).__init__()

        # First block (no pooling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Second block with pooling
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 3 * 3, 32), # 3 for 6 & 7 
            nn.LeakyReLU(),
            # nn.Dropout(0.01),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # print(x.shape)
        x = x.view(-1, 256 * 3* 3)
        x = self.fc_layers(x)
        return x


    
def create_or_update_list_file(filename, number_list):
    """
    Write a list of numbers to a CSV file. Creates a new file if it doesn't exist.
    
    This function is used to save experimental results, optimization data,
    and other numerical data in CSV format for analysis and visualization.

    Args:
        filename (str): The name of the CSV file to write to
        number_list (list): A list of numbers to write to the file
    """
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(number_list)
        
def update_or_create_dataset(file_name, new_X, new_Y):
    """
    Update an existing dataset or create a new one with environment data.
    
    This function manages pickle files containing environment tensors and WCD values,
    allowing for incremental dataset building during data generation and training.
    
    Args:
        file_name (str): Path to the pickle file
        new_X (list): List of new environment tensors to add
        new_Y (list): List of new WCD values to add
    """
    new_X = torch.cat(new_X).cpu()  # Concatenate and move to CPU
    new_Y = torch.tensor(new_Y).cpu()  # Convert to tensor and move to CPU
    file_name = f"{file_name}"
    
    if os.path.exists(file_name):
        # If the file exists, load the existing dataset
        with open(file_name, 'rb') as file:
            dataset = pickle.load(file)
        
        # Update the existing dataset with new data
        dataset.update_data(new_X, new_Y)
    else:
        print("Saved : ", new_X.shape)
        # If the file doesn't exist, create a new dataset
        dataset = CustomDataset(new_X, new_Y)
    
    # # Save the updated or new dataset to the file
    with open(file_name, 'wb') as file:
        pickle.dump(dataset, file)
    
    return dataset    

def decode_env(encoded_env, grid_size=6):
    """
    Convert multi-channel tensor representation back to Overcooked-AI environment layout.
    
    This function is the inverse of encode_env(), converting an 8-channel tensor
    back to a 2D character-based environment layout. It's used for visualization,
    debugging, and interfacing with Overcooked-AI simulation functions.
    
    Channel mapping (reverse of encode_env):
    - Channel 0: Walls ('X')
    - Channel 1: Plates ('P') 
    - Channel 2: Dishes ('D')
    - Channel 3: Onions ('O')
    - Channel 4: Tomatoes ('T')
    - Channel 5: Soups ('S')
    - Channel 6: Empty spaces (' ')
    - Channel 7: Agent positions ('1')
    
    Args:
        encoded_env: 8-channel tensor representation of the environment
        grid_size: Size of the gridworld (default 6)
    
    Returns:
        numpy.ndarray: 2D character array representing the environment layout
    """
    n = encoded_env.shape[1]
    env = np.full((grid_size, grid_size), " ", dtype=str)  # Initialize with empty spaces
    
    # Decode each cell from tensor channels
    for i in range(n):
        for j in range(n):
            # Check each channel in priority order (higher priority channels override lower ones)
            if encoded_env[7, i, j] == 1:  # Agent position (highest priority)
                if env[i, j] == ' ': 
                    env[i, j] = '1'
                
            if encoded_env[3, i, j] == 1:  # Onion
                if env[i, j] == ' ': 
                    env[i, j] = 'O'
            if encoded_env[4, i, j] == 1:  # Tomato
                if env[i, j] == ' ': env[i, j] = 'T'
                
            if encoded_env[1, i, j] == 1:
                if env[i, j] == ' ': env[i, j] = 'P'
                
            if encoded_env[5, i, j] == 1:
                if env[i, j] == ' ': env[i, j] = 'S'
                
            if encoded_env[2, i, j] == 1:
                if env[i, j] == ' ': env[i, j] = 'D'
                
            if encoded_env[6, i, j] == 1:
                if env[i, j] == ' ':env[i, j] = " "
            if encoded_env[0, i, j] == 1:
                if env[i, j] == ' ': env[i, j] = 'X'
            
    return env

def encode_env(env, grid_size=6):
    """
    Convert Overcooked-AI environment layout to multi-channel tensor representation.
    
    This function transforms a 2D character-based environment layout into an 8-channel
    tensor where each channel represents a different object type. This encoding is
    essential for CNN processing and optimization.
    
    Channel mapping:
    - Channel 0: Walls ('X')
    - Channel 1: Plates ('P') 
    - Channel 2: Dishes ('D')
    - Channel 3: Onions ('O')
    - Channel 4: Tomatoes ('T')
    - Channel 5: Soups ('S')
    - Channel 6: Empty spaces (' ')
    - Channel 7: Agent positions (added separately)
    
    Args:
        env: 2D character array representing the environment layout
        grid_size: Size of the gridworld (default 6)
    
    Returns:
        numpy.ndarray: 8-channel tensor representation of the environment
    """
    n = grid_size
    encoded_env = np.zeros((8, n, n))  # Initialize 8-channel tensor
    
    # Encode each cell in the environment
    for i in range(n):
        for j in range(n):
            if env[i, j] == 'X':  # Wall
                encoded_env[0, i, j] = 1
            elif env[i, j] == 'P':  # Plate
                encoded_env[1, i, j] = 1
            elif env[i, j] == 'D':  # Dish
                encoded_env[2, i, j] = 1
            elif env[i, j] == 'O':  # Onion
                encoded_env[3,i, j] = 1
            elif env[i, j] == 'T':
                encoded_env[4,i, j] = 1
            elif env[i, j] == 'S':
                encoded_env[5,i, j] = 1
            elif env[i, j] == " ":
                encoded_env[6,i, j] = 1
            elif env[i, j] == "1":
                encoded_env[7,i, j] = 1
    return encoded_env
    
def encode_multiple_envs(envs,grid_size=6):
    encoded_envs = []
    n = grid_size
    for env_set in envs:
        enc_set = []
        for env in env_set:
            n_rows, n_cols = env.shape
            row_diff = max(0, n - n_rows)
            col_diff = max(0, n - n_cols)
            top_pad = row_diff // 2
            bottom_pad = row_diff - top_pad
            left_pad = col_diff // 2
            right_pad = col_diff - left_pad

            env = np.pad(env, ((top_pad, bottom_pad), (left_pad, right_pad)), constant_values="X")
            
            encoded_env = encode_env(env,grid_size=grid_size)
            enc_set.append(encoded_env)
        

        encoded_envs.append(enc_set)
    return np.stack(encoded_envs)


def read_envs(files):
    # Load environment data from folder
    data_folder = "../data/layouts/"  # Replace with your actual data folder path
    envs = []
    seen_envs = {}
    # Define a regular expression pattern to match the string format "size_<size>_<index>"
    pattern = r"size_(\d+)_(\d+)"
    size_envs ={}
    for filename in files:
        if filename in seen_envs.keys():
            env=seen_envs[filename]
            # envs.append([env,np.flip(env,axis=1)])
            envs.append([env])
            # ,np.flip(env,axis=1)
            continue
        else:
            match = re.match(pattern, filename)

            if match:
                # If there is a match, extract the size and index using the groups method of the match object
                size = int(match.group(1))
                index = int(match.group(2))
                if not size in size_envs.keys():
                    with open(f'../data/layouts/random_generation/envs_layouts_size{size}.pkl', 'rb') as f:
                        size_envs[size] = pickle.load(f)
                env = np.array(size_envs[size][index])
            else:
                layout_name = filename
                over_cookedgridwrld =Mdp.OvercookedGridworld.from_layout_name(layout_name)
                oc_env= Env.OvercookedEnv.from_mdp(over_cookedgridwrld, horizon=100)
                
                env = np.array(over_cookedgridwrld.terrain_mtx)
                player_pos = over_cookedgridwrld.get_standard_start_state().player_positions[0]
                env[player_pos[1]][player_pos[0]]="1"
                seen_envs[filename] =env

        envs.append([env])
        # envs.append([env,np.flip(env,axis=1)])
        #np.flip(env,axis=1)
            
    # Encode multiple environments
    encoded_envs = encode_multiple_envs(envs)
    return encoded_envs

def duplicate_numbers(nums):
    B = nums.shape[0]
    grid = np.zeros((B, 1,6, 6))
    for i in range(B):
        grid[i] = np.full((1,6, 6), nums[i])
    return grid
import numpy as np

def normalize_data(X):
    X[:, :7] = X[:, :7].astype('float32')
    X[:, :7] /= 1.0 # divide by maximum value of 1 (one hot encoded)
    X[:, 7] = X[:, 7].astype('float32')
    X[:, 8] /= 4.0 # normalize second numeric channel to be between 0 and 1
    return X

def print_grid(env):
    for row in env:
        print('  '.join([" " if cell == "" else cell for cell in row]))

def print_env_design(env_layout):
    base_layout_params ={'num_items_for_soup': 3,
                    'all_orders': [{'ingredients': ["tomato",'tomato', 'tomato']}],
                    'recipe_values': [1],
                    'recipe_times': [5.0],
                    'max_num_ingredients': 3}

    over_cookedgridwrld =Mdp.OvercookedGridworld.from_grid(env_layout,base_layout_params)
    oc_env= Env.OvercookedEnv.from_mdp(over_cookedgridwrld, horizon=120)
    statev = StateVisualizer()
    statev.display_rendered_state(oc_env.state,ipython_display=True,grid=over_cookedgridwrld.terrain_mtx)

def env_to_graph(env):
    """Converts the environment to a graph and returns it along with a dictionary of object positions."""
    graph = {}
    objects = {}
    for y, row in enumerate(env):
        for x, node in enumerate(row):
            if node != 'X':
                # Add the node to the graph
                neighbors = []
                if x > 0 and env[y][x-1] != 'X':
                    neighbors.append((y, x-1))
                if x < len(row)-1 and env[y][x+1] != 'X':
                    neighbors.append((y, x+1))
                if y > 0 and env[y-1][x] != 'X':
                    neighbors.append((y-1, x))
                if y < len(env)-1 and env[y+1][x] != 'X':
                    neighbors.append((y+1, x))
                graph[(y, x)] = neighbors
                # Add the node to the objects dictionary if it is an object
                if node != " ":
                    objects[node] = (y, x)
    return graph, objects

def can_traverse(env, node, neighbor,target):
    """Returns True if the agent can traverse from the node to the neighbor in the environment."""
    if neighbor == target: # special case where the neighbor is the target object
        return True 
    y1, x1 = node
    y2, x2 = neighbor
    # Check if the neighbor is an object or X
    if env[y2][x2] != " " and env[y2][x2] != 'X':
        return False
    # Check if the path goes through an object or X
    if y1 == y2:
        for x in range(min(x1, x2)+1, max(x1, x2)):
            if env[y1][x] != " " and env[y1][x] != 'X':
                return False
    elif x1 == x2:
        for y in range(min(y1, y2)+1, max(y1, y2)):
            if env[y][x1] != " " and env[y][x1] != 'X':
                return False
    else:
        return False
    return True

def get_object_positions(env):
    """
    Computes and returns the positions of each object in the environment.

    :param env: A 2D list representing the environment.
    :return: A dictionary mapping each object to its position in the environment.
    """
    object_positions = {}
    rows, cols = len(env), len(env[0])
    for i in range(rows):
        for j in range(cols):
            if env[i][j] != " " and env[i][j] != 'X':
                object_positions[env[i][j]] = (i, j)
    return object_positions


def find_path(env, start, end):
    """Finds a path between the start and end nodes in the environment and returns it."""
    graph, objects = env_to_graph(env)
    visited = set()
    queue = deque([(start, [])])
    # pdb.set_trace()
    while queue:
        node, path = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        if node == end:
            return path + [end]
        for neighbor in graph[node]:
            if can_traverse(env, node, neighbor,end):
                queue.append((neighbor, path + [node]))
    return None

def check_env_is_valid(env):
    if not check_all_objs_reachable(env,from_="P"):
        return False
    elif not check_all_objs_reachable(env,from_="1"):
        return False
    elif not check_env_has_no_empty_edges(env):
        return False
    elif not check_has_single_objects(env):
        return False
    else:
        return True


def check_env_has_no_empty_edges(env):
    # Check first and last rows
    for i in [0, len(env)-1]:
        for j in range(len(env[i])):
            if env[i][j] == " " or env[i][j] == "":
                return False
    
    # Check first and last columns
    for i in range(len(env)):
        for j in [0, len(env[i])-1]:
            if  env[i][j] == " " or  env[i][j] == "":
                return False
            
    return True
def check_all_objs_reachable(env, from_="P",return_binary = True):
    objs_pos = get_object_positions(env) # locate all objects in the environment
    
    # Check if any of the required objects are missing
    objects = ["P", "D", "O", "T", "S","1"]
    for obj in objects:
        if obj not in objs_pos:
            return False if return_binary else 5.0
    
    p_pos = objs_pos[from_]
    if from_ == "P":
        objects.remove("1")
        env = np.where(np.array(env)=="1"," ",np.array(env)).tolist()
        # print(env)
    # Check if all objects are reachable from P
    objects.remove(from_)
    num_unreachable = get_num_unreachable_objects(env,objects,start=p_pos,objs_pos=objs_pos)
    
    return num_unreachable==0 if return_binary  else num_unreachable

def get_num_unreachable_objects(env,objects,start,objs_pos):
    num =0 
    for obj in objects:
        path = find_path(env, start=start, end=objs_pos[obj])
        if path is None:
            num+=1
    return num

def check_has_single_objects(env):
    object_types = {'S', 'P', 'D', 'T', 'O'}
    flat_env = [cell.strip() for row in env for cell in row]
    object_counts = Counter(flat_env)

    multiple_objects = any(object_counts[obj] > 1 for obj in object_types)
    return not multiple_objects

def augment_from_encoded(X,n_augs=4,grid_size=6):
    raw_envs = []
    
    for i in range(0, X.shape[0]):
        env = decode_env(X[i],grid_size=grid_size)
        if n_augs ==4:
            raw_envs.extend([encode_env(env,grid_size=grid_size),encode_env(np.rot90(env),grid_size=grid_size),
                             encode_env(np.rot90(env,2),grid_size=grid_size),
                             encode_env(np.rot90(env,3),grid_size=grid_size),
                             encode_env(np.flip(env,axis=1),grid_size=grid_size)])
        elif n_augs==3:
            raw_envs.extend([encode_env(env,grid_size=grid_size),encode_env(np.rot90(env),grid_size=grid_size),encode_env(np.rot90(env,2),grid_size=grid_size),encode_env(np.rot90(env,3),grid_size=grid_size)])
        elif n_augs==2:
            raw_envs.extend([encode_env(env,grid_size=grid_size),encode_env(np.rot90(env),grid_size=grid_size),encode_env(np.rot90(env,2),grid_size=grid_size)])
        elif n_augs==1:
            raw_envs.extend([encode_env(env,grid_size=grid_size),encode_env(np.rot90(env),grid_size=grid_size)])
        else:
            raw_envs.extend([encode_env(env,grid_size=grid_size)])
    
    return np.array(raw_envs)

def extract_env_details(x,grid_size=6):
    env_layout = decode_env(x.squeeze(),grid_size=grid_size)
    gamma = x[0,8,0,0].item()
    subset_goals = []
    if x[0, 9, 0, 0] == 1:
        subset_goals.append(1)
    if x[0, 10, 0, 0] == 1:
        subset_goals.append(2)
    if x[0, 11, 0, 0] == 1:
        subset_goals.append(3)
    if x[0, 12, 0, 0] == 1:
        subset_goals.append(4)

    return env_layout,gamma,subset_goals



def get_adjacent_spaces(env):
    objects = {}
    for i in range(len(env)):
        for j in range(len(env[0])):
            obj = env[i][j]
            if obj not in [ 'X', " "]:
                adj_spaces = []
                if i > 0 and env[i-1][j] == " ":
                    adj_spaces.append((i-1, j))
                if i < len(env)-1 and env[i+1][j] == " ":
                    adj_spaces.append((i+1, j))
                if j > 0 and env[i][j-1] == " ":
                    adj_spaces.append((i, j-1))
                if j < len(env[0])-1 and env[i][j+1] == " ":
                    adj_spaces.append((i, j+1))
                objects[obj] = adj_spaces
    return objects

from collections import deque




def generate_grids(grid_size, valid_env = True):
    """
    Generate all valid grid arrangements for a given grid size.

    Args:
    - grid_size (int): the size of the square grid (number of rows/columns)

    Returns:
    - valid_grids (list): a list of all valid grid arrangements
    """
    objects = ['P', 'T', 'O', 'S', 'D',"1"]
    spaces = ['X', " "]
    max_duplicates = random.choices([0,1,2,3,4],[0.2,0.2,0.2,0.2,0.2],k=1)[0]
    # max_duplicates = 6
    duplicated_objects = []

    for obj in objects:
        if obj == '1':
            duplicated_objects.append(obj)
        else:
            if valid_env:
                duplicated_objects.append(obj)
            else:
                num_duplicates = random.randint(0, max_duplicates)
                duplicated_objects.extend([obj] * num_duplicates)
    objects = duplicated_objects
            
    # Fix the positions of S, D, and P
    fixed_positions = {
        # 'S': [1, 0],
        # 'D': [0, grid_size - 2],
        # 'P': [grid_size - 2, grid_size // 2]
    }
    
    

    # Dynamically position the remaining objects
    remaining_positions = []
    for i in range(grid_size):
        for j in range(grid_size):
            if [i, j] not in fixed_positions.values():
                remaining_positions.append([i, j])

    # Generate all possible combinations of positions for the remaining objects
    all_combinations = range(100 * grid_size)
    

    # Generate all valid grid arrangements
    generated_grids = []
    for combination in all_combinations:
        # Create a grid with all spaces
        grid = [[" " for _ in range(grid_size)] for _ in range(grid_size)]
        
        for i in range(grid_size):
            for j in range(grid_size):
                if random.random() > 0.6:
                    grid[i][j] = 'X'
                    
        # Add the fixed objects to the grid
        for obj, pos in fixed_positions.items():
            grid[pos[0]][pos[1]] = obj
        

        for i in range(grid_size):
            if not [0,i] in fixed_positions.values():
                grid[0][i] = 'X'
            if not [grid_size-1,i] in fixed_positions.values():
                grid[grid_size-1][i] = 'X'
            if not [i,0] in fixed_positions.values():
                grid[i][0] = 'X'
            if not [i,grid_size-1] in fixed_positions.values():
                grid[i][grid_size-1] = 'X'
        # Add the remaining objects to the grid
        
        picked_ijs = []
        
        
        for i, obj in enumerate(objects):
            if obj not in fixed_positions:
                if obj == "1": #prevent the agent from starting at the boarder
                    i,j = random.randint(1, grid_size-2),random.randint(1, grid_size-2)
                    if (i,j) in picked_ijs or [i,j] in fixed_positions.values():
                        i,j = random.randint(1, grid_size-2),random.randint(1, grid_size-2)
                else:
                    i,j = random.randint(0, grid_size-1),random.randint(0, grid_size-1)
                    if (i,j) in picked_ijs or [i,j] in fixed_positions.values():
                        i,j = random.randint(0, grid_size-1),random.randint(0, grid_size-1)
                picked_ijs.append((i,j))
                
                grid[i][j] = obj
        
        if valid_env:
            if check_env_is_valid(grid):
                generated_grids.append(grid)
        else:
            if not check_env_is_valid(grid):
                generated_grids.append(grid)

    return generated_grids

