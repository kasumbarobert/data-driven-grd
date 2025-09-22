
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

import random
from torch.utils.data import Dataset

import pickle

import os
import re
import torch.nn as nn
import csv
import gpytorch

from torchvision.models import resnet50, resnet18,resnet34,resnet101,vgg16
from sklearn.kernel_ridge import KernelRidge
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data,DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, TopKPooling

RANDOM_SEED = 42

def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(RANDOM_SEED)


from einops import rearrange

class VisionTransformer(nn.Module):
    def __init__(self, n_channels=13, height=64, width=64, patch_size=8, embed_dim=8, num_heads=4, num_layers=6, drop_out=0.1):
        """
        Vision Transformer (ViT) with default parameters.

        Args:
            n_channels (int, optional): Number of input channels. Default is 13.
            height (int, optional): Input height. Default is 64.
            width (int, optional): Input width. Default is 64.
            patch_size (int, optional): Size of each image patch. Default is 8.
            embed_dim (int, optional): Dimension of patch embeddings. Default is 128.
            num_heads (int, optional): Number of attention heads. Default is 4.
            num_layers (int, optional): Number of transformer encoder layers. Default is 6.
            drop_out (float, optional): Dropout rate. Default is 0.1.
        """
        super(VisionTransformer, self).__init__()

        # Calculate the number of patches
        self.num_patches = (height // patch_size) * (width // patch_size)
        self.patch_size = patch_size

        # Patch embedding: Flatten patches and project to embed_dim
        self.patch_embed = nn.Conv2d(
            n_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        self.dropout = nn.Dropout(drop_out)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=drop_out
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        self.fc1 = nn.Linear(embed_dim, 32)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(0.0001)

        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(drop_out)

        self.fc3 = nn.Linear(16, 1)
        self.reluN = nn.ReLU()

    def forward(self, x):
        # Create patches and flatten them
        x = self.patch_embed(x)  # Shape: (batch_size, embed_dim, num_patches_h, num_patches_w)
        x = rearrange(x, "b c h w -> b (h w) c")  # Flatten patches

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.dropout(x)

        # Pass through transformer
        x = self.transformer(x)  # Shape: (batch_size, num_patches, embed_dim)

        # Aggregate the patch embeddings (mean pooling)
        x = x.mean(dim=1)

        # Pass through output layers
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.reluN(self.fc3(x))
        return x

class GNNModel(nn.Module):
    def __init__(self, n_channels=4, hidden_dim=32, output_dim=1, drop_out=0.01, num_heads=4):
        super(GNNModel, self).__init__()
         # Graph Attention layers (GAT)
        # Graph Attention layers (GAT)
        self.gat1 = GATConv(n_channels, hidden_dim, heads=num_heads, dropout=drop_out)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=drop_out)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * num_heads, 128)  # Adjusted input size after GAT layers
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(drop_out)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim * num_heads)  # Should match the output size of GAT layers
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # First GAT layer
        x = self.gat1(x, edge_index)
        x = self.relu(x)
        
        # Apply batch normalization after GAT1 layer
        x = self.bn1(x)  # Apply BatchNorm1d on node features
        x = self.dropout(x)

        # Second GAT layer
        x = self.gat2(x, edge_index)
        x = self.relu(x)
        
        # Apply global pooling (mean pooling over all nodes in the graph)
        x = global_mean_pool(x, batch)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn2(x)  # Apply BatchNorm1d on the fully connected output
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.fc3(x)  # Output layer
        
        return x


class LinearModel(nn.Module):
    def __init__(self, n_channels=13, height=64, width=64):
        """
        True linear model: a single linear mapping from input to output.
        
        Args:
            n_channels (int): Number of input channels.
            height (int): Height of the input.
            width (int): Width of the input.
        """
        super(LinearModel, self).__init__()
        
        # Compute the flattened input size
        self.input_size = n_channels * height * width

        # Single linear transformation
        self.linear = nn.Linear(self.input_size, 1)

    def forward(self, x):
        # Flatten the input tensor (batch_size, n_channels, H, W) -> (batch_size, flattened_size)
        x = x.view(x.size(0), -1)

        # Apply the linear layer
        x = self.linear(x)
        return x


class KernelRidgeRegression(nn.Module):
    def __init__(self, input_dim, alpha=1.0, kernel="rbf", gamma=None):
        super(KernelRidgeRegression, self).__init__()
        
        # Kernel Ridge Regression from sklearn
        self.krr = KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma)
        
        # Input preprocessing (e.g., flattening)
        self.flatten = nn.Flatten()
        self.input_dim = input_dim

    def fit(self, X, y):
        """
        Train the Kernel Ridge Regression model.
        Args:
            X: Input data (features).
            y: Target values.
        """
        X_flat = X.view(-1, self.input_dim).detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        self.krr.fit(X_flat, y)

    def forward(self, x):
        """
        Perform inference using the trained Kernel Ridge Regression model.
        Args:
            x: Input data (features).
        Returns:
            Predictions as a PyTorch tensor.
        """
        x_flat = self.flatten(x).detach().cpu().numpy()
        y_pred = self.krr.predict(x_flat)
        return torch.tensor(y_pred, dtype=torch.float32).to(x.device)

class GaussianProcessRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GaussianProcessRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


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
    if not ("data" in file_name or "grid" in file_name): #else use file name
        file_name =f"./data/{file_name}"
        
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

def convert_x_to_graph(tensor):
    """
    Converts a single tensor (n_channels x 13 x 13) to a graph-compatible dictionary.

    Args:
        tensor (torch.Tensor): Input tensor of shape (n_channels, 13, 13).

    Returns:
        graph_tensor (dict): A dictionary containing:
            - 'node_features': Tensor of node features (num_nodes, n_channels).
            - 'edge_index': Tensor of edge indices (2, num_edges).
    """
    n_channels, grid_size, _ = tensor.shape
    assert grid_size == 13, "The input grid size must be 13x13."

    node_features = []
    edge_index = []
    node_map = {}  # Map (i, j) -> node ID

    # Step 1: Create nodes and extract features
    node_id = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if tensor[0, i, j] == 1:  # Only add free spaces as nodes
                # Map grid cell (i, j) to a node ID
                node_map[(i, j)] = node_id
                node_id += 1

                # Extract node features from all channels
                features = tensor[:, i, j].tolist()  # [space, blocked, start, goal, ...]
                node_features.append(features)

    # Step 2: Create edges based on adjacency
    for (i, j), node_id in node_map.items():
        for ni, nj in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:  # Up, down, left, right
            if (ni, nj) in node_map:  # Ensure the neighbor is a valid node
                edge_index.append([node_id, node_map[(ni, nj)]])
    
    # Convert edge_index and node_features to tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    node_features = torch.tensor(node_features, dtype=torch.float)

    # Create graph dictionary for this instance
    graph_tensor = {
        'node_features': node_features,  # Node features tensor
        'edge_index': edge_index,       # Adjacency information
    }

    return graph_tensor



def create_graph_dataset(batched_tensor, targets):
    """
    Converts a batched tensor and corresponding targets into a PyTorch Geometric dataset.
    
    Args:
        batched_tensor (torch.Tensor): Shape (B, n_channels, 13, 13).
        targets (torch.Tensor): Shape (B,).
    
    Returns:
        list[Data]: A list of PyTorch Geometric Data objects.
    """
    graph_dataset = []
    for i in range(batched_tensor.shape[0]):
        graph_tensor = convert_x_to_graph(batched_tensor[i])  # Use the sub-function
        graph_data = Data(
            x=graph_tensor['node_features'],        # Node features
            edge_index=graph_tensor['edge_index'],  # Edge indices
            y=targets[i]                            # Target label
        )
        graph_dataset.append(graph_data)
    return graph_dataset




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

def bfs_shortest_distances(grid_size, start, goal, blocked):
    # Initialize an nxn matrix to store distances
    distances = [[float('inf')] * grid_size for _ in range(grid_size)]
    distances[goal[0]][goal[1]] = 0  # Distance to goal is 0

    # Queue for BFS
    queue = deque([goal])
    
    # Define the possible movements (up, down, left, right)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # Perform BFS
    while queue:
        current_x, current_y = queue.popleft()

        for dx, dy in directions:
            next_x, next_y = current_x + dx, current_y + dy
            
            # Check if next position is within the grid
            if 0 <= next_x < grid_size and 0 <= next_y < grid_size:
                # Check if the next position is not blocked
                if (next_x, next_y) not in blocked:
                    # Calculate distance from goal to next position
                    distance_to_next = distances[current_x][current_y] + 1
                    # Update the distance if it's smaller
                    if distance_to_next < distances[next_x][next_y]:
                        distances[next_x][next_y] = distance_to_next
                        queue.append((next_x, next_y))

    # Update blocked positions to have distance as the square of the grid size
    for blocked_x, blocked_y in blocked:
        distances[blocked_x][blocked_y] = grid_size ** 2

    return distances


def compute_wcd_single_env_no_paths(grid_size, goal_positions, blocked_positions, start_pos, vis_paths = False, return_paths = False):
    """
    Computes the WCD for a single gridworld environment.
    
    :param grid_size: Size of the gridworld.
    :param goal_positions: List of tuples representing goal positions.
    :param blocked_positions: List of tuples representing blocked positions.
    :param start_pos: Tuple representing the start position.
    :return: WCD value for the environment.
    """
    if not is_design_valid(grid_size, goal_positions, blocked_positions, start_pos)[0]: # invalid env
        return None 
        
    paths_to_goals = []
    dist_matrix_to_goals = []

    for i,goal_pos in enumerate(goal_positions):
        temp_blocked_positions = blocked_positions+[goal_positions[1-i]] # the other goal's  position is blocked
        shortest_path_len = bfs_shortest_distances(grid_size, start=start_pos, goal=goal_pos, blocked=temp_blocked_positions)
        dist_matrix_to_goals.append(shortest_path_len)
        # plot_grid_with_numbers(np.array(shortest_path_len))
    # Compute WCD based on paths_to_goals
    
    goal1_paths = np.array(dist_matrix_to_goals[0])
    goal2_paths = np.array(dist_matrix_to_goals[1])
    
    temp_blocked_positions = blocked_positions + goal_positions + [start_pos] # the other goal's  position is blocked
    shortest_path_len = bfs_shortest_distances(grid_size, start=None, goal=start_pos, blocked=temp_blocked_positions)
    dist_matrix_to_goals.append(shortest_path_len)
    goal3_paths = np.array(dist_matrix_to_goals[2])
    goal3_paths[start_pos[0],start_pos[1]] = 0
    
    if goal1_paths[start_pos[0],start_pos[1]] != goal2_paths[start_pos[0],start_pos[1]]:
        goal2_paths = goal2_paths+(goal1_paths[start_pos[0],start_pos[1]]-goal2_paths[start_pos[0],start_pos[1]])
    
    # print(goal1_paths[start_pos[0],start_pos[1]], goal2_paths[start_pos[0],start_pos[1]])
    a = goal1_paths[start_pos[0],start_pos[1]]
    idx = np.logical_and(goal1_paths==goal2_paths, goal1_paths + goal3_paths == a)
    wcd = a - np.min(goal1_paths[idx])
    # wcd = goal1_paths[start_pos[0],start_pos[1]]-np.min(goal1_paths[goal1_paths==goal2_paths])
    # goal1_paths==goal2_paths)
    # return goal1_paths, goal2_paths, goal3_paths, wcd
    return wcd


def compute_true_wcd_no_paths(x):
    grid_size, goal_positions, blocked_positions, start_pos,space_pos = decode_grid_design(x)
    return compute_wcd_single_env_no_paths(grid_size, goal_positions, blocked_positions, start_pos,vis_paths= False)

def randomize_pos(grid_size, num_goal_pos, num_special_pos, num_blocked_pos):
    local_random = random.Random()
    # Check if the total number of positions exceeds the grid capacity
    if num_goal_pos + num_special_pos + num_blocked_pos >= grid_size**2:
        raise ValueError("Total number of positions exceeds the capacity of the grid.")

    all_possible_pos = [(i, j) for i in range(grid_size) for j in range(grid_size)]

    # Define possible positions for goal positions in the last columns
    last_column_pos = [(i, j) for i in range(grid_size) for j in range(grid_size - 2, grid_size)]
    new_goal_pos = local_random.sample(last_column_pos, num_goal_pos)
    
    remaining_pos = list(set(all_possible_pos) - set(new_goal_pos))

    # Define possible positions for special reward positions in the first three columns
    first_three_column_pos = list(set([(i, j) for i in range(grid_size) for j in range(2,6)]) & set(remaining_pos))
    new_special_reward_pos = local_random.sample(first_three_column_pos, num_special_pos)
    remaining_pos = list(set(remaining_pos) - set(new_special_reward_pos))
    
    # Ensure there are enough remaining positions for blocked positions
    if len(remaining_pos) < num_blocked_pos + 1: # +1 to account for the start position
        raise ValueError("Not enough remaining positions for blocked positions after allocating goal and special reward positions.")

    # Randomly select unique positions for blocked positions
    new_blocked_pos = local_random.sample(remaining_pos, num_blocked_pos)
    remaining_pos = list(set(remaining_pos) - set(new_blocked_pos))

    # Choose start position from the first column of the remaining positions
    first_column_pos = [(i, 0) for i in range(grid_size) if (i, 0) in remaining_pos]
    if not first_column_pos:
        raise ValueError("No available positions in the first column for the start position.")
    start_pos = local_random.choice(first_column_pos)

    return new_goal_pos, new_special_reward_pos, new_blocked_pos, start_pos


def encode_grid_design(n, goal_positions, blocked_positions, start_pos):
    # Creating a 3D array with 4 channels, each of size n x n
    channels = np.zeros((4, n, n))

    # Marking the spaces (everything not blocked is a space)
    channels[0, :, :] = 1
    for blocked in blocked_positions:
        channels[0, blocked[0], blocked[1]] = 0  # Marking blocked cells as 0 in spaces channel

    # Marking the blocked spaces
    for blocked in blocked_positions:
        channels[1, blocked[0], blocked[1]] = 1
        channels[0, blocked[0], blocked[1]] = 0 # mark this as blocked in the space channel

    # Marking the starting position
    channels[2, start_pos[0], start_pos[1]] = 1
    channels[0, start_pos[0], start_pos[1]] = 0 # mark this as blocked in the space channel

    # Marking the goal positions
    for goal in goal_positions:
        channels[3, goal[0], goal[1]] = 1
        channels[0, goal[0], goal[1]] = 0 # mark this as blocked in the space channel

    return torch.tensor(channels).unsqueeze(0)

def extract_positions(grid, get_space_pos = False):
    start_pos = None
    blocked_positions = []
    goal_positions = []
    space_positions = []
    grid_size = len(grid)
    for x, row in enumerate(grid):
        for y, cell in enumerate(row):
            if cell == 'S':
                start_pos = (x, y)
            elif cell == 'X':
                blocked_positions.append((x, y))
            elif cell == 'G':
                goal_positions.append((x, y))
            elif cell == ' ':
                space_positions.append((x, y))
    if get_space_pos:
        return grid_size, goal_positions, blocked_positions, start_pos, space_positions
    else:   
        return grid_size, goal_positions, blocked_positions, start_pos

def encode_from_grid_to_x(grid):
    grid_size, goal_positions, blocked_positions, start_pos = extract_positions(grid)
    return encode_grid_design(grid_size, goal_positions, blocked_positions, start_pos)


def is_design_valid(grid_size, goal_positions, blocked_positions, start_pos):
    shortest_path_lens =[]
    for i,goal_pos in enumerate(goal_positions):
        temp_blocked_positions = blocked_positions+[goal_positions[1-i]]
        shortest_path_len = bfs_shortest_path(grid_size, start=start_pos, goal=goal_pos, blocked=temp_blocked_positions)
        shortest_path_lens.append(shortest_path_len)
            # return False  # Invalid environment, as there's no path to a goal
    return not (-1 in shortest_path_lens), shortest_path_lens

def decode_grid_design(encoded_grid, return_map = False):
    # Ensure that the input is a numpy array for easy handling
    if isinstance(encoded_grid, torch.Tensor):
        encoded_grid = encoded_grid.numpy()
    
    # Extract the channels
    spaces, blocked, start, goals = encoded_grid

    # Initialize an empty grid for visualization
    n = spaces.shape[1]  # Assuming square grid
    grid = np.full((n, n), ' ')
   

    # Mark start and goal positions
    start_pos = np.argwhere(start == 1) # the list
    goal_pos = np.argwhere(goals == 1)
    blocked_pos = np.argwhere(blocked == 1)
    spaces_pos = np.argwhere(spaces == 1)
    
    start_position = tuple(start_pos.tolist()[0]) # the single point
    
    if not return_map: # extract the positions 
        goal_pos =  [ tuple(pos) for pos in goal_pos.tolist()]
        blocked_pos = [tuple(pos) for pos in blocked_pos.tolist() if tuple(pos) != start_position]
        spaces_pos = [tuple(pos) for pos in spaces_pos.tolist() if tuple(pos) != start_position]
        return n, goal_pos, blocked_pos ,start_position ,spaces_pos
    
     # Mark blocked positions
    for i in range(n):
        for j in range(n):
            if blocked[i, j] == 1:
                grid[i, j] = 'X'
    if len(start_pos) > 0:
        grid[start_pos[0, 0], start_pos[0, 1]] = 'S'
    
    for i,pos in enumerate(goal_pos):
        grid[pos[0], pos[1]] = f'G{i}'
    
    return grid


def compute_wcd_single_env(grid_size, goal_positions, blocked_positions, start_pos, vis_paths = False, return_paths = False):
    """
    Computes the WCD for a single gridworld environment.
    
    :param grid_size: Size of the gridworld.
    :param goal_positions: List of tuples representing goal positions.
    :param blocked_positions: List of tuples representing blocked positions.
    :param start_pos: Tuple representing the start position.
    :return: WCD value for the environment.
    """
    if not is_design_valid(grid_size, goal_positions, blocked_positions, start_pos)[0]: # invalid env
        return None 
        
    paths_to_goals = []
    for i,goal_pos in enumerate(goal_positions):
        temp_blocked_positions = blocked_positions+[goal_positions[1-i]] # the other goal's  position is blocked
        shortest_path_len = bfs_shortest_path(grid_size, start=start_pos, goal=goal_pos, blocked=temp_blocked_positions)
        if shortest_path_len == -1:
            return None  # Invalid environment, as there's no path to a goal
        paths = find_shortest_paths(grid_size, start_pos, goal_pos, shortest_path_len, temp_blocked_positions)
        paths_to_goals.append(paths)
        
        if vis_paths:
            mdp = GridworldMDP(n=grid_size, goal_state_pos=[goal_pos], goal_state_rewards=[5.2], blocked_pos=blocked_positions, start_pos=start_pos)
            # print("WCD=",compute_wcd_from_paths([paths[0]],[paths[1]]))
            for path in paths:
                print("goal",goal_pos)
                mdp.visualize(path)
    # Compute WCD based on paths_to_goals
    wcd,wcd_paths = compute_wcd_from_paths(paths_to_goals[0],paths_to_goals[1],return_wcd_paths = True)
    
    if return_paths:
        return wcd, paths_to_goals, wcd_paths
        
    return wcd


def compute_true_wcd(x):
    grid_size, goal_positions, blocked_positions, start_pos,space_pos = decode_grid_design(x)
    return compute_wcd_single_env(grid_size, goal_positions, blocked_positions, start_pos,vis_paths= False)

def plot_grid(grid):
    # Map each character to a color
    color_map = {
        ' ': 'white',  # Empty space
        'X': 'white',  # Blocked
        'G': 'green',  # Goal
        'S': 'orange'    # Start
    }

    # Map each character to an integer
    int_map = {char: i for i, char in enumerate(color_map.keys())}

    # Convert grid to an array of integers using int_map
    int_grid = np.array([[int_map[cell] for cell in row] for row in grid])

    # Create a ListedColormap from color_map
    cmap = ListedColormap(color_map.values())

    # Plotting
    fig, ax = plt.subplots()
    ax.matshow(int_grid, cmap=cmap)

    # Adding grid labels
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            c = grid[i][j]
            ax.text(j, i, str(c), va='center', ha='center', color='red' if c in ['X', ' '] else 'white')
    ax.set_xticks(np.arange(-.5, len(grid[0]), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(grid), 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    # ax.set_xticks([])
    # ax.set_yticks([])
    plt.show()
    
def check_design_is_valid(x):
    grid_size, goal_positions, blocked_positions, start_pos,space_pos = decode_grid_design(x)
    return is_design_valid(grid_size, goal_positions, blocked_positions, start_pos)


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