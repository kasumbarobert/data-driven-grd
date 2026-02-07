"""Converted from human_exp.ipynb.
Generated automatically by tools/notebook_to_script.py.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import pandas as pd
import itertools
import math
from collections import deque
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import ast
import copy
import os

# %% [markdown]
# ## train model from read human data

# %% [code] cell 1
# from mdp import GridworldMDP

# from utils import *

# %% [code] cell 4
Actions = []
human_exp_actions = [(-1, 0), (1,0), (0,-1), (0,1)]
# case "ArrowUp": case 'W': case 'w':
#     movePlane({ row: -1, col: 0 }, action=0);
#     break;
# case "ArrowDown": case 'S': case 's':
#     movePlane({ row: 1, col: 0 }, action=1);
#     break;
# case "ArrowLeft": case 'A': case 'a':
#     movePlane({ row: 0, col: -1 }, action=2);
#     break;
# case "ArrowRight": case 'D': case 'd':
#     movePlane({ row: 0, col: 1 }, action=3);
#     break;

# %% [markdown]
# ## read human dataset

# %% [code] cell 6
data = pd.read_csv('human-data/formal1-200.csv')
actions = data['Answer.output_user_actions']
gameidx = data['Answer.output_gameidx']
gamelayouts = np.load('human-data/layout_data3274.npz')['x']

# %% [code] cell 7
actions_len = actions.apply(lambda x:[len(k) for k in json.loads(x)])

# %% [code] cell 8
actions = data[['WorkerId','Answer.output_user_actions']]
actions['move'] = actions['Answer.output_user_actions'].apply(lambda x:json.loads(x))
# actions.head()
actions = actions[['WorkerId', 'move']].explode('move')
actions.head(17), len(actions )
actions.index = np.arange(3200)
idx = np.arange(15, 3200, 16)
print(idx[:10])
actions.drop(idx, inplace=True)
actions.head(17), len(actions )

# %% [code] cell 9
gameidx = data[['WorkerId', 'Answer.output_gameidx']].explode('Answer.output_gameidx')
gameidx['idx'] = gameidx['Answer.output_gameidx'].apply(lambda x: ast.literal_eval(x))
gameidx = gameidx[['WorkerId', 'idx']].explode('idx')
gameidx.head(), len(gameidx)

# %% [code] cell 10
gameidx.index = np.arange(3000)
actions.index = np.arange(3000)
actions['idx'] = gameidx['idx']
print(actions.shape)
del data

# %% [code] cell 11
print(len(gamelayouts), gamelayouts[0], type(gamelayouts[0]))

# %% [markdown]
# ## generate human data

# %% [code] cell 13
np.random.seed(42)
useridx = np.arange(200)
np.random.shuffle(useridx)
train_idx = useridx[:160]
val_idx = useridx[160:180]
test_idx = useridx[180:]

# %% [code] cell 14
simulation_actions = [(1,0), (-1,0), (0,1), (0,-1)] ## in simulations
human_exp_actions = [(-1, 0), (1,0), (0,-1), (0,1)]

# %% [code] cell 15
## drop those data is len > 19
actions['len'] = actions['move'].apply(lambda x: len(x))
print(sum(actions['len']>=19))
print(sum(actions['len']<=2))

# %% [code] cell 16
## below is to check whether to actions and layout map are consistent
def visualize_human_action(move, idx):
    layout = np.copy(gamelayouts[idx])
    quick_visualize(layout)
    print(move)
    for a in move:
        change = human_exp_actions[a]
        print(a, change, '---'*10)
        i,j = np.argwhere(layout[2,:,:])[0]
        layout[2, i,j] = 0
        layout[2, i+change[0], j+change[1]] = 1
        quick_visualize(layout)
    return -1

def quick_visualize(layout):
    ans = np.zeros([6,6], dtype='object')
    for i in range(6):
        for j in range(6):
            ans[i][j] = '0'
            if layout[1,i,j]==1:
                ans[i][j] = 'X'
            if layout[2,i,j]==1:
                ans[i][j] = 'S'
            if layout[3,i,j]==1:
                ans[i][j] = 'G'
    for inner_list in ans:
        print(inner_list)

visualize_human_action(actions.iloc[0]['move'], actions.iloc[0]['idx'])

# %% [code] cell 17
grid_size = 6

def get_action(pos, newpos):
    move = (int(newpos[0]-pos[0]), int(newpos[1]-pos[1]))
    # print(move)
    if move in simulation_actions:
        return simulation_actions.index(move)
    else:
        return int(4)

def rotate(pos, grid_size):
    if type(pos) is tuple:
        return (grid_size-1-pos[1], pos[0])
    else:
        return [(grid_size-1-p[1], p[0]) for p in pos]

def human_data_gene(move, idx):
    layout = gamelayouts[idx]

    envs = []
    ans = [] ## the output actions
    if len(move) >= 19:
        return None, None

    goal = tuple(np.argwhere(layout[3,:,:])[0])
    new_blocked_positions = np.argwhere(layout[1,:,:])
    cur_pos = tuple(np.argwhere(layout[2,:,:])[0])
    for a in move:
        change = human_exp_actions[a]

        envs.append(encode_grid_design(grid_size, [goal], new_blocked_positions, cur_pos))
        ans.append(simulation_actions.index(change))

        tmp_goal = goal
        tmp_block = new_blocked_positions
        tmp_pos = cur_pos
        tmp_next = (cur_pos[0] + change[0], cur_pos[1] + change[1])

        for _ in range(3):
            tmp_goal = rotate(tmp_goal, grid_size)
            tmp_block = rotate(tmp_block, grid_size)
            tmp_pos = rotate(tmp_pos, grid_size)
            tmp_next = rotate(tmp_next, grid_size)
            envs.append(encode_grid_design(grid_size, [tmp_goal], tmp_block, tmp_pos))
            ans.append(get_action(tmp_pos, tmp_next))

        cur_pos = (cur_pos[0] + change[0], cur_pos[1] + change[1])

    tensor_y = torch.tensor(ans, dtype=torch.int)
    tensor_x = torch.cat(envs).type(torch.int)
    return tensor_x, tensor_y

x, y = human_data_gene(actions.iloc[0]['move'], actions.iloc[0]['idx'])

# %% [code] cell 18
userx = []
usery = []

for i in range(200):
    userx.append([])
    usery.append([])
    for j in range(15):
        k = i * 15 + j
        x, y = human_data_gene(actions.iloc[k]['move'], actions.iloc[k]['idx'])
        if y is not None:
            userx[i].append(x)
            usery[i].append(y)
    userx[i] = torch.cat(userx[i])
    usery[i] = torch.cat(usery[i])

# %% [code] cell 19
def get_data(idx):
    x,y =  torch.cat([userx[i] for i in idx ]), torch.cat([usery[i] for i in idx ] )
    print(x.shape, y.shape)
    return x,y

trainX, trainY = get_data(train_idx)
valX, valY = get_data(val_idx)
testX, testY = get_data(test_idx)

# %% [code] cell 20
torch.save((trainX, trainY, valX, valY, testX, testY), 'human-data/formal1-train.pt')

# %% [markdown]
# ## train the human model

# %% [code] cell 22
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %% [code] cell 23
# data = torch.load('human-data/formal1-train.pt')
data = torch.load('human-data/formal-train-grid10.pt')

for i in range(6):
    print(data[i].shape)

trainX, trainY, valX, valY, testX, testY = data[0], data[1], data[2], data[3], data[4], data[5]
del data

np.random.seed(23)
idx = np.arange(len(trainX))
np.random.shuffle(idx)
trainX = trainX[idx, :,:,:]
trainY = trainY[idx]
torch.unique(trainY, return_counts = True)

trainX = trainX.to(device)
trainY = trainY.to(device)
valX = valX.to(device)
valY = valY.to(device)
testX = testX.to(device)
testY = testY.to(device)

# %% [code] cell 24

class HumanNN(nn.Module):
    def __init__(self, batch_size=2**6, batch_num=100000, num_workers=8,lr=1e-6, traindata_path=None, valdata_path=None, fc1_size = 512, state_save_path = '', reg=0):
        super(HumanNN, self).__init__()
        self.fc1_size = fc1_size
        self.fc1 = nn.Linear(400, self.fc1_size)
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

# %% [code] cell 25
t1 = time.time()
batch_size = 4125
num = len(trainX)
batch_num = num//batch_size

for fc_size in [512, 1024]:
    for lr in [0.0003, 0.001]:
        for momentum in [0.0]:
            for l2 in [0.0]:
                if os.path.exists(f'human-model-formal2/model_grid10_{num}_{fc_size}_{lr}_{momentum}_{l2}.pt'):
                    continue

                model = HumanNN(fc1_size=fc_size).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=momentum, weight_decay=l2)
                max_iter = 4000
                records = np.zeros([max_iter, 3]) # training loss/test loss/acc
                for i in range(max_iter):
                    for batch in range(batch_num):
                        optimizer.zero_grad()
                        pred = model.forward(trainX[batch*batch_size:(batch+1)*batch_size])
                        loss = criterion(pred, trainY[batch*batch_size:(batch+1)*batch_size].type(dtype=torch.long))
                        loss.backward()
                        optimizer.step()
                    records[i,0] = loss.item()
                    if i % 20 == 0:
                        with torch.no_grad():
                            pred = model.forward(valX)
                            pred_label = torch.argmax(pred, dim=1)
                            acc = torch.sum((pred_label == valY))/(len(pred_label))
                            loss_test = criterion(pred, valY.type(dtype=torch.long))
                            records[i,1:3] = [loss_test.item(), acc.item()]
                    if i % 1000 == 0:
                        print(f'iter={i}, training loss={loss.item()}, testing_loss={loss_test.item()}, accuracy={acc} ,time={time.time()-t1}')

                torch.save(model, f'human-model-formal2/model_grid10_{num}_{fc_size}_{lr}_{momentum}_{l2}.pt')
                np.savez(f'human-model-formal2/training_{num}_{fc_size}_{lr}_{momentum}_{l2}.npz', record=records)
                print('finish ', lr, fc_size, momentum, np.max(records[:,2]))
                print('---'*10)

# %% [code] cell 26
torch.save(model, f'human-model-formal1/model_grid6_{num}_{fc_size}_{lr}_{momentum}_{l2}.pt')
np.savez(f'human-model-formal1/training_{num}_{fc_size}_{lr}_{momentum}_{l2}.npz', record=records)
print('finish ', lr, fc_size, momentum, np.max(records[:,2]))
print('---'*10)

# %% [code] cell 27
os.path.exists(f'human-model-formal1/model_grid6_{num}_{fc_size}_{lr}_{momentum}_{l2}.pt')

# %% [markdown]
# ## evaluate the model

# %% [code] cell 30
def evaluation(model, X, Y):
    with torch.no_grad():
        pred = model.forward(X)
        pred_label = torch.argmax(pred, dim=1)
        acc = torch.sum((pred_label == Y))/(len(Y))
    return acc


for fc_size in [512, 1024]:
    for lr in [0.0003,0.001]:
        for momentum in [0.0]:
            for l2 in [0.0]:
                model = torch.load(f'human-model-formal2/model_grid10_{num}_{fc_size}_{lr}_{momentum}_{l2}.pt').to( device)
                ans = [evaluation(model, trainX, trainY),
                      evaluation(model, valX, valY),
                      evaluation(model, testX, testY)]
                print(fc_size, lr, momentum, ans)

# %% [code] cell 31
torch.device('cpu')
model = model.to(torch.device('cpu'))
testX = testX.to(torch.device('cpu'))
testY = testY.to(torch.device('cpu'))

# %% [code] cell 32
Actions = [(1,0), (-1,0), (0,1), (0,-1)]

def get_action(pos, newpos):
    move = (int(newpos[0]-pos[0]), int(newpos[1]-pos[1]))
    # print(move)
    if move in Actions:
        return Actions.index(move)
    else:
        return int(4)

def move_action(pos, move):
    return (pos[0] + move[0], pos[1] + move[1])

def filter_action(layout):
    layout = layout.numpy()
    start = tuple(np.argwhere(layout[2,:,:])[0])
    end = tuple(np.argwhere(layout[3,:,:])[0])
    block = layout[1,:,:]
    valid_action = [False] * 4
    for k, move in enumerate(Actions):
        newpos = move_action(start, move)
        if newpos[0]<0 or newpos[0]>=10 or newpos[1]<0 or newpos[1]>=10:
            valid_action[k] = False
        elif block[newpos[0], newpos[1]] == 1:
            valid_action[k] = False
        else:
            valid_action[k] = True

    return get_action(start, end), valid_action

test_num = len(testX)
action_flag = np.zeros(test_num)
valid_action = np.zeros((test_num,4))

for i in range(test_num):
    action_flag[i], valid_action[i,:] = filter_action(testX[i,:,:,:])

print(np.unique(action_flag, return_counts = True))
print(np.unique(np.sum(valid_action, axis=1), return_counts = True))


# (array([1., 2., 3., 4.]), array([ 44, 339, 928, 689]))

# %% [code] cell 33
def filter_accuracy(y_test, pred, action_flag, valid_action):
    pred_label = torch.argmax(pred, dim=1)
    acc0 = torch.sum((pred_label == y_test))/(len(y_test))
    pred_filter = pred * torch.tensor(valid_action)
    pred_label = torch.argmax(pred_filter, dim=1)
    acc1 = torch.sum((pred_label == y_test))/(len(y_test))
    action_flag = torch.tensor(action_flag).to(dtype=torch.int64)
    pred_label[action_flag!=4] = action_flag[action_flag!=4]
    acc2 = torch.sum((pred_label == y_test))/(len(y_test))
    return acc0, acc1, acc2

pred = model.forward(testX)
filter_accuracy(testY, pred, action_flag, valid_action)

# %% [markdown]
# ## get the path prediction
