import torch
import random
import random
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import ast
import copy
import os
from torch.distributions.categorical import Categorical
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

Actions = [(1,0), (-1,0), (0,1), (0,-1)]
grid_size = 6


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
    

def filter_action(layout):
    if type(layout) is torch.Tensor:
        layout = layout.reshape(4,grid_size,grid_size).numpy()
    else:
        layout = layout.reshape(4,grid_size,grid_size)
    start = tuple(np.argwhere(layout[2,:,:])[0])  
    end = tuple(np.argwhere(layout[3,:,:])[0])  
    block = layout[1,:,:]
    valid_action = [False] * 4
    for k, move in enumerate(Actions):
        newpos = move_action(start, move) 
        if newpos[0]<0 or newpos[0]>=grid_size or newpos[1]<0 or newpos[1]>=grid_size:
            valid_action[k] = False
        elif block[newpos[0], newpos[1]] == 1:
            valid_action[k] = False
        else:
            valid_action[k] = True
    print(start, valid_action)
    return get_action(start, end), valid_action

def filter_action_pos(cur_pos, block):
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

def get_action(pos, newpos):
    move = (int(newpos[0]-pos[0]), int(newpos[1]-pos[1]))
    # print(move)
    if move in Actions:
        return Actions.index(move)
    else:
        return int(4)

def decode_grid_design(encoded_grid, return_map = False):
    # Ensure that the input is a numpy array for easy handling
    if isinstance(encoded_grid, torch.Tensor):
        encoded_grid = encoded_grid.numpy()
    # Extract the channels
    spaces, blocked, start, goals = encoded_grid
    # Initialize an empty grid for visualization
    n = spaces.shape[1]  # Assuming square grid
    grid = np.full((n, n), '   ')
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

def get_human_prob(model, grid_size, goal_positions, blocked_positions, start_pos, moves):
    designs = []
    for i,goal_pos in enumerate(goal_positions):
        temp_blocked_positions = blocked_positions +[goal_positions[1-i]] # the other goal's  position is blocked
        # temp_blocked_positions = blocked_positions
        # print(blocked_positions, temp_blocked_positions,temp_blocked_positions)
        designs.append(encode_grid_design(grid_size, [goal_pos], temp_blocked_positions, start_pos))
    ans = []
    # print(designs)
    for i, layout in enumerate(designs):
        prob = compute_human_probs(layout, model, moves)
        ans.append(prob)
    # return softmax(ans, beta = 0.03) ## this is used in test to adjust the number
    return softmax(ans)

def compute_human_probs(layout, model, moves):
    ## move is the action in current state 
    Actions = [(1,0), (-1,0), (0,1), (0,-1)]
    # print(layout.shape)
    # layout = designs[0][0,:,:,:]
    block = layout[0,1,:,:].numpy().astype(int)
    # print(block)
    max_step = 20  # fail to reach goal if reach max_step
    cur_layout = layout.reshape([1,4,grid_size,grid_size]).detach().clone()
    cur_pos = tuple(torch.argwhere(cur_layout[0,2,:,:]).detach().numpy()[0])
    prob = 0
    for i in range(len(moves)):
        # human_pred = model.forward(cur_layout).detach().numpy().reshape(-1)
        human_pred = model.forward_beta(cur_layout, beta=0.01).detach().numpy().reshape(-1)
        # human_pred = human_pred - np.min(human_pred)
        ## get valid actions
        for k, a in enumerate(Actions):
            newpos = move_action(cur_pos, a)
            if newpos[0] < 0 or newpos[0] >= 6 or newpos[1] < 0 or newpos[1] >= 6:
                human_pred[k] = 0
            elif block[newpos[0], newpos[1]] == 1:
                human_pred[k] = 0
            # print(newpos, human_pred[k])
        human_pred = human_pred / np.sum(human_pred)
        
        prob += np.log(human_pred[moves[i]])
        # print(i,cur_pos,  human_pred)
        a = moves[i]
        cur_layout[0,2,cur_pos[0], cur_pos[1]] = 0
        cur_layout[0,0,cur_pos[0], cur_pos[1]] = 1
        cur_pos = (cur_pos[0] + Actions[int(a)][0], cur_pos[1] + Actions[int(a)][1])
        if cur_pos[0] < 0 or cur_pos[0] >= grid_size or cur_pos[1] < 0 or cur_pos[1] >= grid_size:
            break
        cur_layout[0,2,cur_pos[0], cur_pos[1]] = 1
        cur_layout[0,0,cur_pos[0], cur_pos[1]] = 0
    return prob


def softmax(x, beta = 1.0):
    y = np.exp((np.array(x) - np.max(x)) * beta )
    return y/np.sum(y)


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

def move_action(pos, move):
    return (pos[0] + move[0], pos[1] + move[1]) 

