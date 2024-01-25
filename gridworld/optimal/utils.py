
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import torch
import pandas as pd
import itertools
import math
# from mdp import GridworldMDP

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import torch
import pandas as pd
import itertools
from itertools import product,combinations
import math

import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from collections import deque

import random
from torch.utils.data import Dataset

import pickle

import os
import re
import torch.nn as nn
import csv

import json


with open(f'{os.path.dirname(os.path.realpath(__file__))}/config.json', 'r') as config_file:
    config = json.load(config_file)

if config["USE_GPU_DEVICE"]=="yes":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = "cpu"

print("DEVICE set to", DEVICE)

from torchvision.models import resnet50, resnet18,resnet34,resnet101,vgg16

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
        

        



class GridworldMDP:
    def __init__(self, n, goal_state_pos=None, goal_state_rewards=None, blocked_pos=[], start_pos=None, special_reward_pos=[], special_rewards=None):
        """
        Initializes the Gridworld MDP (Markov Decision Process).

        Parameters:
        - n (int): The size of the gridworld (n x n).
        - goal_states (list of tuples): Positions of the goal states in the grid.
        - goal_state_rewards (list of floats): Rewards for reaching each goal state.
        - blocked_pos (list of tuples): Positions of the blocked states in the grid.
        - start_pos (tuple): The starting position of the agent in the grid.
        - special_reward_states (list of tuples): Positions of states with special rewards.
        - special_rewards (list of floats): Rewards for reaching each special reward state.
        """

        # Initialize class variables
        self.n = n  # Grid size
        self.init_pos = start_pos  # Initial position of the agent
        self.goal_state_pos= goal_state_pos if goal_state_pos is not None else []  # Goal states
        # self.goal_states = [(pos, True) for pos in self.goal_state_pos] # the goal state is only after a subgoal is reached
        self.goal_state_rewards = goal_state_rewards if goal_state_rewards is not None else []  # Rewards for goal states
        self.agent_pos = start_pos  # Current position of the agent
        self.special_reward_pos = special_reward_pos if special_reward_pos is not None else []  # States with special rewards
        # self.special_reward_states = [(pos, True) for pos in self.special_reward_pos] # the goal state is only after a subgoal is reached
        self.special_rewards = special_rewards if special_rewards is not None else []  # Special rewards
        self.blocked_pos = blocked_pos  # Blocked states
        self.states = self.generate_states()  # Generate all possible states
        
                
        self.actions = ['left', "stay", 'up', 'down', 'right']  # Possible actions
        self.transitions = self._generate_transitions()  # Generate state transitions
        self.rewards = self._generate_rewards()  # Generate rewards for state transitions
        self.visited_sub_goals = [] # no state is visited
        self.cumulative_reward = 0.0
        

    def _generate_transitions(self):
        """
        Generates the transition probabilities for each state-action pair.

        Returns:
        - P (list): A 3D list containing transition probabilities.
        """
        P = [None]*len(self.states)
        for s in range(len(self.states)):
            P[s] = [None]*len(self.actions)
            for a in range(len(self.actions)):
                P[s][a] = [0]*len(self.states)
                next_state=self._next_state(self.states[s], self.actions[a], p=1.0)
                next_state_idx = self.states.index(next_state)
                
                if self.states[s][0] in self.goal_state_pos and next_state[0] not in self.goal_state_pos:  # prevent leaving goal state
                    P[s][a][next_state_idx] =  0.0
                elif self.states[s][0] in self.blocked_pos:
                    P[s][a][next_state_idx] = 0.0
                elif self.states[next_state_idx][0] in self.blocked_pos:
                    P[s][a][next_state_idx] = 0.0
                else:
                    P[s][a][next_state_idx] = 1.0
        return P

    def get_transitions(self):
        """
        Returns the transition probabilities.

        Returns:
        - self.transitions (list): The transition probabilities.
        """
        return self.transitions

    def _generate_rewards(self):
        """
        Generates the rewards for each state-action pair in a Markov Decision Process.
        Rewards are assigned based on the state and action taken, with special consideration for goal states, special reward states, and subgoals.

        Returns:
        - R (list): A 3D list containing rewards for each state-action-next state triplet.
        """
        # Initialize the rewards list
        R = [None]*len(self.states)
        for s in range(len(self.states)):
            R[s] = [None]*len(self.actions)
            for a in range(len(self.actions)):
                R[s][a] = [0]*len(self.states)

                # Calculate the next state based on the current state and action
                s_prime = self._next_state(self.states[s], self.actions[a], p=1)
                # print(s_prime)
                # if s_prime[0] in self.special_reward_pos:
                #     print(s_prime[0] in s_prime[1], s_prime[0], s_prime[1])
                if s_prime[0] in s_prime[1]: # already visited
                    R[s][a][self.states.index(s_prime)] = 0.0 # Zero reward for going to an already visited state
                elif self.states[s][0] in self.goal_state_pos:  # If in a goal state
                    # Penalize any action other than 'stay'
                    R[s][a][self.states.index(s_prime)] = 0 # penalize leaving the goal state

                elif self.states[s][0] in self.special_reward_pos:  # If in a goal state
                    # Penalize any action other than 'stay'
                    if self.actions[a] == "stay":
                        R[s][a][self.states.index(s_prime)] = 0
                        
                elif s_prime[0] in self.goal_state_pos:  # If the next state is a goal state
                    # Assign reward based on reaching the goal state
                    R[s][a][self.states.index(s_prime)] = self.goal_state_rewards[self.goal_state_pos.index(s_prime[0])]

                elif s_prime[0] in self.special_reward_pos:  # If the next state has a special reward
                    # Assign a special reward for reaching this state
                    R[s][a][self.states.index(s_prime)] = self.special_rewards[self.special_reward_pos.index(s_prime[0])]

                elif not s_prime[1]:  # If the subgoal has not been reached in the next state
                    # Assign a negative reward to encourage reaching subgoals
                    R[s][a][self.states.index(s_prime)] = 0.0 #-0.01

                else:  # For all other cases
                    # Assign -0.1 for staying and  no reqard for moving
                    if self.actions[a] == "stay":
                        R[s][a][self.states.index(s_prime)] = -0.01 # staying in a non-goal state should be punished
        
        return R


    def _next_state(self, state, action, p=1):
        """
        Computes the next state given a current state and an action.

        Parameters:
        - state (tuple): The current state.
        - action (str): The action to be taken.
        - p (float): Probability of the action (default is 1).

        Returns:
        - next_state (tuple): The next state after performing the action.
        """
        (i, j), visited_subgoals = state
        sub_goal_status = state[1]
        
        if action == 'up':
            next_i = max(i-1, 0)
            next_j = j
        elif action == 'down':
            next_i = min(i+1, self.n-1)
            next_j = j
            
        elif action == 'left':
            next_i = i
            next_j = max(j-1, 0)
        elif action == 'right':
            next_i = i
            next_j = min(j+1, self.n-1)
        elif action == 'stay':
            next_i = i
            next_j = j
        else:
            raise ValueError("Invalid action")
        
        
        next_pos = (next_i, next_j)
        visited_subgoals = visited_subgoals.copy()
        # print(visited_subgoals)
        if (i,j) in self.special_reward_pos and (i,j) not in visited_subgoals:
            visited_subgoals.append((i,j))
            visited_subgoals = sorted(visited_subgoals)
        
        if (i,j) in self.goal_state_pos: #once a goal state is reached all rewards are 0
            next_pos = (i,j)
            visited_subgoals = sorted(self.special_reward_pos + self.goal_state_pos )
            
        if next_pos in self.blocked_pos:
            return (i, j), visited_subgoals # update this state to show that (i,j) is visited if it's a subgoal
        
        next_state = (next_pos,visited_subgoals)
        
        return next_state

    def reset(self):
        """
        Resets the agent to the initial position.

        Returns:
        - self.agent_pos (tuple): The reset position of the agent.
        """
        self.agent_pos = self.init_pos
        self.visited_sub_goals = [] # NO state visited so far
        return (self.agent_pos,self.visited_sub_goals)
    

    def move(self, action):
        """
        Moves the agent according to the given action.

        Parameters:
        - action (str): The action to be taken.

        Returns:
        - self.agent_pos (tuple): The new position of the agent.
        """
        curr_state = (self.agent_pos,self.visited_sub_goals)
        next_state = self._next_state(curr_state, action, p=1)
        self.agent_pos,self.visited_sub_goals= next_state
        reward = self.get_rewards()[self.states.index(curr_state)][self.actions.index(action)][self.states.index(next_state)] # extract the reward from the reward table
        
        self.cumulative_reward+=reward
        # print("to",next_state[0],"reward",reward)
        return self.agent_pos,self.visited_sub_goals, reward
    
    def get_cumulative_reward(self):
        return self.cumulative_reward
    
    def generate_all_subgoal_visit_combinations(self):
        """
        Generates all possible state matrices with varying combinations of True and False.

        Returns:
        - subgoals_visited_combinations : List of subgoals_visited_combinations.
        """
    
        
        subgoals_visited_combinations =  [
                sorted(list(combination))
                for r in range(0, len(self.special_reward_pos+self.goal_state_pos)+1)
                for combination in combinations(self.special_reward_pos+self.goal_state_pos, r)
            ]
        return subgoals_visited_combinations

    def generate_states(self):
        """
        Generates all possible states in the gridworld.

        Returns:
        - all_states (list of tuples): All possible states in the grid.
        """
        state_visited = [True,False]
        
        all_states = []
        subgoal_visit_combinations = self.generate_all_subgoal_visit_combinations()

        
        
        for i in range(self.n):
            for j in range(self.n):
                for is_visited in subgoal_visit_combinations:
                    if (i,j) not in self.blocked_pos : # these will always be unreachable
                        all_states.append(((i, j),is_visited))
        
        # print("There are ", len(all_states)," states")
        return all_states
                                

    def get_rewards(self):
        """
        Returns the rewards.

        Returns:
        - self.rewards (list): The rewards.
        """
        return self.rewards

    def get_states(self):
        """
        Returns the states.

        Returns:
        - self.states (list of tuples): The states.
        """
        return self.states

    def get_curr_state_index(self):
        """
        Returns the index of the current state of the agent.

        Returns:
        - Index (int): The index of the current state.
        """
       
        
        return self.states.index((self.agent_pos,self.visited_sub_goals))

    def get_state_index(self, state):
        return self.states.index(state)
        
    
    def action_index(self, action):
        if action in self.actions:
            return self.actions[index]
        else:
            raise ValueError('Invalid action')
    
    def index_action(self, index):
        index = int (index)
        if index < len(self.actions): 
            return self.actions[index]
        else:
            raise ValueError('Invalid action index')
    def get_mdp_representation(self):
        return {
        "grid_size":self.n,
        "blocked_positions": self.blocked_pos,
        "start_pos": self.init_pos,
        "sub_goal_positions": self.special_reward_pos,
        "sub_goal_rewards": self.special_rewards,
        "goal_positions": self.goal_state_pos,
        "goal_rewards": self.goal_state_rewards,
        "gamma": 1
    }
        
    def visualize(self, path=None):
        
        grid = construct_grid(self.get_mdp_representation())
        plot_grid(grid,path)
        
        
    def computeQFunction(self, gamma, T = 50):
        Nstate = len(self.get_states())
        Naction = len(self.actions)
        T = T
        R = torch.tensor(self.get_rewards())
        P = torch.tensor(self.get_transitions())
        policy = torch.zeros([ T, Nstate])
        Vs = torch.zeros([T, Nstate])

        Q = torch.zeros([T, Nstate, Naction])
        R = torch.sum((R*P), dim=2)
        # print(R)
        for step in range (T):
            pos = T  - step - 1
            if pos == T -1: # greedy in last step
                q_f = R
                policy[pos,:] = torch.argmax(q_f, dim=1)
                Q[pos, :, : ] = q_f
                Vs[pos,:] = torch.max(q_f,axis=1)[0]
            else: # optimal in expectation
                q_f = R  + gamma * torch.sum(P[:, :, :] * (  Vs[pos + 1, :]), axis=2) 
                # + torch.randn(R.shape) * (1e-4)
                # + torch.randn(R.shape) * (1e-5)
                Q[pos, :, :] = q_f
                policy[pos,:] = torch.argmax(q_f,dim=1)
                Vs[pos,:] = torch.max(q_f,axis=1)[0]

        df =pd.DataFrame(zip(self.get_states(),torch.argmax(Q[0],dim=1).numpy(),Q[0].numpy()))
        pd.set_option('display.max_rows', None)


        return policy.cpu()[0],Q[0].cpu()



def construct_grid(decoded_info):
    # Initialize the grid
    grid_size = decoded_info["grid_size"]
    grid = np.full((grid_size, grid_size)," ",dtype='U10')

    # Mark blocked positions
    blocked_positions = decoded_info["blocked_positions"]
    for pos in blocked_positions:
        grid[pos[0], pos[1]] = "X"

    # Mark start position
    start_pos = decoded_info["start_pos"]
    grid[start_pos[0], start_pos[1]] = "S"

    # Mark sub-goal positions and rewards
    sub_goal_positions = decoded_info["sub_goal_positions"]
    sub_goal_rewards = decoded_info["sub_goal_rewards"]
    for pos, reward in zip(sub_goal_positions, sub_goal_rewards):
        grid[pos[0], pos[1]] = f"{reward}"
        print(grid[pos[0], pos[1]], reward)

    # Mark goal positions and rewards
    goal_positions = decoded_info["goal_positions"]
    goal_rewards = decoded_info["goal_rewards"]
    for pos, reward in zip(goal_positions, goal_rewards):
        grid[pos[0], pos[1]] = f"{reward}"

    return grid
    
def plot_grid(grid, path=None):
    # Map each character to a color
    color_map = {
        ' ': 'white',   # Empty space
        'X': 'white',   # Blocked
        'S': 'orange'   # Start
    }

    # Define colors for values below and above 1
    color_below_1 = 'lightblue'
    color_above_1 = 'lightgreen'

    # Plotting
    fig, ax = plt.subplots()

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            c = grid[i][j]

            # Assign colors based on the value
            if c in ['X', ' ', 'S']:
                color = color_map[c]
            elif float(c) < 1:
                color = color_below_1
            else:
                color = color_above_1

            # Display the cell with the assigned color
            ax.add_patch(plt.Rectangle((j, len(grid) - i - 1), 1, 1, fill=True, edgecolor='black', facecolor=color))
            ax.text(j + 0.5, len(grid) - i - 0.5, str(c), va='center', ha='center', color='red' if c in ['X', ' '] else 'black')

    if path:
        for pos in path:
            ax.add_patch(plt.Rectangle((pos[1], len(grid) - pos[0] - 1), 1, 1, fill=True, edgecolor='black', facecolor='orange'))

    # Setting axis limits
    ax.set_xlim(0, len(grid[0]))
    ax.set_ylim(0, len(grid))

    # Set axis labels
    ax.set_xticks(np.arange(0.5, len(grid[0]) + 0.5, 1))
    ax.set_yticks(np.arange(0.5, len(grid) + 0.5, 1))
    ax.set_xticklabels(np.arange(0, len(grid[0]), 1))
    # ax.set_yticklabels(np.arange(0, len(grid), 1))
    ax.set_yticklabels(np.arange(len(grid) - 1, -1, -1))
    plt.show()
    
