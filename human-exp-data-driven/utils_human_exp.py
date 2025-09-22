
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import torch
import pandas as pd
import itertools
import math
# from mdp import GridworldMDP

from collections import deque

import random
from torch.utils.data import Dataset

import pickle

import os
import re
import torch.nn as nn
import csv
import torch.nn as nn
import torch.nn.functional as F

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
    
class CustomCNN(nn.Module):
    def __init__(self, n_channels=13,drop_out = 0.01,size = 3):
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
        
        # # Second block with pooling
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(256, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU()
        # )


        self.conv_output_size = 256 * size*size
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.conv_output_size, 16), # 3 for 6 & 7 
            nn.LeakyReLU(),
            # nn.Dropout(0.01),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(8, 1),
            nn.ReLU()
        )
        

    def forward(self, x):
        x = self.conv1(x)
        # pdb.set_trace()
        x = self.conv2(x)
        # x = self.conv3(x)
        # print(x.shape)
        x = x.view(-1, self.conv_output_size)
        x = self.fc_layers(x)
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
    if not "data" in file_name: #else use file name
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
    
    if len(goal_positions) == 0 or start_pos is None:
        return False, []
    
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
    return compute_wcd_single_env(grid_size, goal_positions, blocked_positions, start_pos,model = model)


def compute_humal_model_wcd(x, model= None,search_depth = 19):
    grid_size, goal_positions, blocked_positions, start_pos,space_pos = decode_grid_design(x)
    return compute_human_wcd(model,grid_size, goal_positions, blocked_positions, start_pos,search_depth=search_depth)

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
    
def check_design_is_valid(x,human_model=None, is_grid = True):
    if is_grid:
        grid_size, goal_positions, blocked_positions, start_pos = extract_positions(x)
    else:
        grid_size, goal_positions, blocked_positions, start_pos,space_pos = decode_grid_design(x)
        
    
    is_valid, other_ = is_design_valid(grid_size, goal_positions, blocked_positions, start_pos)

    if not is_valid:
        return is_valid, other_
    
#     wcd = compute_human_wcd(grid_size, goal_positions, blocked_positions, start_pos,model = human_model)
    
#     if wcd is None:
#         return False, other_
    
    return is_valid, other_ 


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



Actions = [(1,0), (-1,0), (0,1), (0,-1)]

    


def low_stochasticity_argmax(human_pred, epsilon=1e-6):
    # Compute softmax probabilities
    softmax_probs = np.exp(human_pred) / np.sum(np.exp(human_pred))
    
    # Adjust probabilities to introduce low stochasticity
    adjusted_probs = softmax_probs * (1 - epsilon)
    adjusted_probs[np.argmax(human_pred)] += epsilon
    
    # Use np.random.choice to sample from the predictions array based on adjusted probabilities
    chosen_index = np.random.choice(np.arange(len(human_pred)), p=adjusted_probs)
    
    return chosen_index
    


def compute_human_wcd(model, grid_size, goal_positions, blocked_positions, start_pos, search_depth =19):
    layout0 = encode_grid_design(grid_size, goal_positions, blocked_positions, start_pos).squeeze().numpy()
    all_layout = generate_all_layout(layout0,grid_size)
    valids, human_pred = move_pred(all_layout, model,grid_size)
    goal_pos1 = tuple(np.argwhere(layout0[3,:,:])[0])
    goal_pos2 = tuple(np.argwhere(layout0[3,:,:])[1])

    move1, wcd1 = compute_human_path_searchk_preset(layout0, all_layout, valids, model, look_ahead = search_depth, grid_size = grid_size, goal_pos = goal_positions[0])
    move2, wcd2 = compute_human_path_searchk_preset(layout0, all_layout, valids, model, look_ahead = search_depth, grid_size = grid_size, goal_pos = goal_positions[1])

    if ~wcd1 and ~wcd2:
        y = find_wcd(move1, move2)
        return y
    else:
        return -10 # invalid




Actions = [(1,0), (-1,0), (0,1), (0,-1)]


class HumanNN(nn.Module):
    def __init__(self, batch_size=2**6, batch_num=100000, num_workers=8,lr=1e-6, traindata_path=None, valdata_path=None, fc1_size = 512, state_save_path = '', reg=0):
        super(HumanNN, self).__init__()
        self.fc1_size = fc1_size
        self.fc1 = nn.Linear(144, self.fc1_size) if grid_size == 6 else nn.Linear(400, self.fc1_size) ## grid_size = 10
        self.fc2 = nn.Linear(self.fc1_size,self.fc1_size)
        self.fc3 = nn.Linear(self.fc1_size, self.fc1_size)
        self.fc4 = nn.Linear(self.fc1_size, 4)
        self.sm = nn.Softmax(dim=1)
        
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.num_workers = num_workers  
        self.lr = lr
        self.traindata_path = traindata_path
        self.valdata_path = valdata_path
        self.iter_count = 0
        self.state_save_path = state_save_path
        self.reg = reg ### ewgularization part  
        
    def forward(self,x):
        out = F.relu(self.fc1(torch.flatten(x, start_dim=1).type(torch.float)))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.sm(self.fc4(out))
        return out
    
    def forward_beta(self, x, beta = 0.03):
        out = F.relu(self.fc1(torch.flatten(x, start_dim=1).type(torch.float)))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.sm(self.fc4(out) * float(beta))
        return out
        
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y, y_hat)
        celoss = -torch.mean(y * torch.log(y_hat)) 
        argloss = 0
        self.log("performance", {"iter": batch_idx, "loss": loss, "CEloss": celoss, "meanbeta":0, 'argloss':argloss, 'beta':0})
        return celoss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.iter_count += 1 
        y_hat = self.forward(x)
        loss = self.loss_fn(y, y_hat)
        celoss = -torch.mean(y * torch.log(y_hat))
        self.log("performance", {"iter": batch_idx, "val_loss": loss, "Cess": celoss, "meanbeta":0, 'argloss':0, 'beta':0})
        
        if (self.iter_count %10 == 0 ):
            torch.save(self.state_dict(), self.state_save_path + '_'+ str(self.iter_count) + '.pt')   
        return celoss

    def loss_fn(self, y, pred):
        rmsel = nn.MSELoss()
        return rmsel(y, pred)
    
    def train_dataloader(self):
        if os.path.exists(self.traindata_path):
            data = torch.load(self.traindata_path)
            x, y = data['human_x'], data['human_y']
            if self.batch_size * self.batch_num < x.size()[0]:
                x, y = x[:self.batch_size * self.batch_num], y[:self.batch_size * self.batch_num]
        else:
            print('check train data path:', self.traindata_path)
            sys.exit(0)
        if self.data_augment:
            ds = torch.utils.data.TensorDataset(x,y)
            dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return dl
    
    def val_dataloader(self):
        if os.path.exists(self.valdata_path):
            data = torch.load(self.valdata_path)
            x, y = data['human_x'], data['human_y']
        else:
            print('check validation data path:', self.valdata_path)
            sys.exit(0)
        ds = torch.utils.data.TensorDataset(x,y)
        dl = torch.utils.data.DataLoader(ds, batch_size=x.size()[0], shuffle=False, num_workers=1)    
        return dl


def compute_path_dis_model(layout, dis, grid_size = 6):
    Actions = [(1,0), (-1,0), (0,1), (0,-1)]
    invmove = [1,0,3,2]
    max_step = 20 if grid_size == 6 else 30 # fail to reach goal if reach max_step
    cur_layout = np.copy(layout)
                        #  .reshape([1,4,grid_size,grid_size]))
    cur_pos = tuple(np.argwhere(cur_layout[2,:,:])[0])
    move = []
    flag = False
    goal_pos = [tuple(x) for x in np.argwhere(cur_layout[3,:,:])]
    for t in range(max_step):
        action_flag, valid_action = filter_action_pos(cur_pos, cur_layout[1],grid_size)
        if len(move)> 0:
            valid_action[invmove[move[-1]]] = False ## so cannot return to the next action 
        human_pred = [20] * 4
        for i, m in enumerate(Actions):
            if valid_action[i]:
                new_pos = move_action(cur_pos, m)
                human_pred[i] = dis[new_pos[0], new_pos[1]]
        human_pred = -np.array(human_pred)
        if sum(valid_action) == 0: # no action available
            flag = True
            break
        a = np.argmax(human_pred)
        move.append(int(a))
        cur_pos = (cur_pos[0] + Actions[int(a)][0], cur_pos[1] + Actions[int(a)][1])
        if cur_pos[0] < 0 or cur_pos[0] >= grid_size or cur_pos[1] < 0 or cur_pos[1] >= grid_size:
            break
        if cur_pos in goal_pos:
            break
    return move, (len(move) >= max_step) | flag

def build_map(layout, start_pos):
    dis = np.ones((grid_size,grid_size), dtype = int) * 100
    dis[start_pos[0], start_pos[1]] = 0
    # blocked = layout[1].numpy().astype(int)
    blocked = layout[1]
    Actions = [(1,0), (-1,0), (0,1), (0,-1)]
    distance = 0
    candidate_list = [start_pos]
    while len(candidate_list) >= 1 and distance <=20:  ## make sure no dead lock, and ignore rate cases where distance might be > 20
        for i, j in candidate_list:
            ## assign value 
            for a in Actions:
                m, n = i+a[0], j+a[1]
                if m>=0 and m<grid_size and n>=0 and n<grid_size:
                    if blocked[m,n] == 0:
                        dis[m, n] = min (1+distance ,  dis[m, n])
        distance += 1
        candidate_list = list(np.argwhere(dis == distance))
        # print(len(candidate_list), distance)
    return dis

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

def filter_action_pos(cur_pos, block,grid_size):
    valid_action = [False] * 4
    for k, move in enumerate(Actions):
        newpos = move_action(cur_pos, move) 
        if newpos[0]<0 or newpos[0]>=grid_size or newpos[1]<0 or newpos[1]>=grid_size:
            valid_action[k] = False
        elif block[newpos[0], newpos[1]] == 1:
            valid_action[k] = False
        else:
            valid_action[k] = True
    return -1, valid_action

def encode_grid_design_numpy(n, goal_positions, blocked_positions, start_pos):
    # Creating a 3D array with 4 channels, each of size n x n
    channels = np.zeros((4, n, n))

    # Marking the spaces (everything not blocked is a space)
    channels[0, :, :] = 1
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
    return channels

def generate_all_layout(layout, grid_size):
    # layout.shape # 4,6,6
    all_layout = np.zeros([grid_size,grid_size,4,grid_size,grid_size])
    for i, j in itertools.product(range(grid_size), range(grid_size)):
        tmp_layout = np.zeros([4,grid_size,grid_size])
        tmp_layout[0] = layout[0] + layout[1]
        tmp_layout[1] = layout[1]
        tmp_layout[3] = layout[3]
        tmp_layout[2,i,j] = 1
        tmp_layout[0,i,j] = 0
        all_layout[i,j] = tmp_layout
    return torch.tensor(all_layout)
        
def move_pred(layouts, model,grid_size):
    human_pred = model.forward(layouts.reshape(-1,4,grid_size,grid_size)).detach().numpy().reshape(grid_size,grid_size,4)
    valids = np.zeros([grid_size,grid_size,4])
    for i, j in itertools.product(range(grid_size), range(grid_size)):
        action_flag, valid_action = filter_action(layouts[i,j],grid_size)
        valids[i,j] = valid_action
    human_pred = human_pred * valids
    return valids, human_pred

def get_valids(layouts):
    valids = np.zeros([grid_size,grid_size,4])
    for i, j in itertools.product(range(grid_size), range(grid_size)):
        action_flag, valid_action = filter_action(layouts[i,j],grid_size)
        valids[i,j] = valid_action
    return valids 

def move_pred_preset(layouts, model, grid_size,valids):
    human_pred = model.forward(layouts.reshape(-1,4,grid_size,grid_size)).detach().numpy().reshape(grid_size,grid_size,4)
    human_pred = human_pred * valids
    return human_pred

def compute_human_path_searchk(layout, model, look_ahead = 0, grid_size = 6):
    Actions = [(1,0), (-1,0), (0,1), (0,-1)]
    all_layout = generate_all_layout(layout,grid_size)
    valids, human_pred = move_pred(all_layout, model,grid_size)
    goal_pos = tuple(np.argwhere(layout[3,:,:])[0])
    blocks = layout[1]
    # layout = designs[0][0,:,:,:]
    max_step = 20 if grid_size == 6 else 30# fail to reach goal if reach max_step
    cur_pos = tuple(np.argwhere(layout[2,:,:])[0])
    # goal_pos = tuple(torch.argwhere(cur_layout[0,3,:,:]).detach().numpy()[0])
    move = []
    state_history = [cur_pos]
    flag = False
    for _ in range(max_step):
        f, a = searchk_action(blocks, cur_pos, goal_pos, valids, human_pred, state_history, look_ahead)
        if not f:
            flag = True
            break
        move.append(a)
        cur_pos = (cur_pos[0] + Actions[int(a)][0], cur_pos[1] + Actions[int(a)][1])
        if cur_pos[0] < 0 or cur_pos[0] >= grid_size or cur_pos[1] < 0 or cur_pos[1] >= grid_size:
            flag = True
            break
        # print(cur_pos, goal_pos)
        state_history.append(cur_pos)
        if cur_pos[0] == goal_pos[0] and cur_pos[1] == goal_pos[1]:
            break
    # print(state_history)
    return move, (len(move) >= max_step) | flag

def searchk_action(blocks, cur_pos, goal_pos, valids, human_pred, state_history, look_ahead):
    ## return whether reach the goal or is valid, action 
    i,j = cur_pos[0], cur_pos[1]
    if blocks[i,j] == 1 or np.sum(valids[i,j]) == 0:
        return False, -1
    actions = np.argsort( -human_pred[i,j] )
    for a in actions:
        if valids[i,j,a]:
            tmp_pos = (cur_pos[0] + Actions[int(a)][0], cur_pos[1] + Actions[int(a)][1])
            if tmp_pos == goal_pos:
                return True, a ## reach the goal 
            if tmp_pos in state_history:
                continue ## skip current action if already exists
            if look_ahead == 0 and tmp_pos not in state_history:
                return True, a ## no look ahead, and action is feasible
            else: ## need look ahead
                f, b = searchk_action(blocks, tmp_pos, goal_pos, valids, human_pred, state_history + [tmp_pos], look_ahead - 1)
                if f:
                    return True, a
                else:
                    continue ## conitnue to search            
        else:
            return False, -1
    return False, -1

def filter_action(layout, human=False,grid_size=6):
    if type(layout) is not np.ndarray:
        layout = layout.numpy()
    start = tuple(np.argwhere(layout[2,:,:])[0])  
    end = tuple(np.argwhere(layout[3,:,:])[0])  
    block = layout[1,:,:]
    valid_action = [False] * 4
    actions = Actions
    for k, move in enumerate(actions):
        newpos = move_action(start, move) 
        if newpos[0]<0 or newpos[0]>=grid_size or newpos[1]<0 or newpos[1]>=grid_size:
            valid_action[k] = False
        elif block[newpos[0], newpos[1]] == 1:
            valid_action[k] = False
        else:
            valid_action[k] = True
    return get_action(start, end), valid_action

def get_action(pos, newpos):
    move = (int(newpos[0]-pos[0]), int(newpos[1]-pos[1]))
    # print(move)
    if move in Actions:
        return Actions.index(move)
    else:
        return int(4)
    
def move_action(pos, move):
    return (pos[0] + move[0], pos[1] + move[1])


def compute_human_path_searchk_preset(layout, all_layout, valids,  model, look_ahead = 0, grid_size = 6, goal_pos = (0,0)):
    Actions = [(1,0), (-1,0), (0,1), (0,-1)]
    human_pred = move_pred_preset(all_layout, model,grid_size, valids)
    blocks = layout[1]
    max_step = 20 if grid_size == 6 else 30# fail to reach goal if reach max_step
    cur_pos = tuple(np.argwhere(layout[2,:,:])[0])
    move = []
    state_history = [cur_pos]
    flag = False
    for _ in range(max_step):
        # action without duplicate records 
        f, a = searchk_action(blocks, cur_pos, goal_pos, valids, human_pred, state_history, look_ahead)
        if not f:
            flag = True
            break
        # a = int(np.argmax(human_pred[cur_pos[0], cur_pos[1]]))
        move.append(a)
        cur_pos = (cur_pos[0] + Actions[int(a)][0], cur_pos[1] + Actions[int(a)][1])
        if cur_pos[0] < 0 or cur_pos[0] >= grid_size or cur_pos[1] < 0 or cur_pos[1] >= grid_size:
            flag = True
            break
        # print(cur_pos, goal_pos)
        state_history.append(cur_pos)
        if cur_pos[0] == goal_pos[0] and cur_pos[1] == goal_pos[1]:
            break
    # print(state_history)
    return move, (len(move) >= max_step) | flag


def searchk_action(blocks, cur_pos, goal_pos, valids, human_pred, state_history, look_ahead):
    ## return whether reach the goal or is valid, action 
    i,j = cur_pos[0], cur_pos[1]
    if blocks[i,j] == 1 or np.sum(valids[i,j]) == 0:
        return False, -1
    actions = np.argsort( -human_pred[i,j] )
    for a in actions:
        if valids[i,j,a]:
            tmp_pos = (cur_pos[0] + Actions[int(a)][0], cur_pos[1] + Actions[int(a)][1])
            if tmp_pos == goal_pos:
                return True, a ## reach the goal 
            if tmp_pos in state_history:
                continue ## skip current action if already exists
            if look_ahead == 0 and tmp_pos not in state_history:
                return True, a ## no look ahead, and action is feasible
            else: ## need look ahead
                f, b = searchk_action(blocks, tmp_pos, goal_pos, valids, human_pred, state_history + [tmp_pos], look_ahead - 1)
                if f:
                    return True, a
                else:
                    continue ## conitnue to search            
        else:
            return False, -1
    return False, -1

def find_wcd(path1, path2):
    a = 0
    for k in range(min(len(path1), len(path2))):
        if path1[k] == path2[k]:
            a += 1
        else:
            break
    return int(a) 

