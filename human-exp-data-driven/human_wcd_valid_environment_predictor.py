"""Converted from Human WCD Valid Environment Predictor .ipynb.
Generated automatically by tools/notebook_to_script.py.
"""

import sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso,BayesianRidge, LogisticRegression,SGDClassifier
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR, SVC
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import torch
import ast
from joblib import dump, load
from sklearn.pipeline import Pipeline
import pickle
from IPython.display import Markdown
import os
import re
import torch.nn as nn
import random
import cProfile
import torch.optim as optim
import torch.nn.utils as utils
import wandb
import pdb
from utils_human_exp import *
from utils import *
from torch.utils.data import DataLoader, TensorDataset

# %% [code] cell 0

plt.figure(dpi=150)

# %% [code] cell 2
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
GRID_SIZE = 6

# wandb.login(relogin=True)
# os.environ['WANDB_API_KEY'] = "71f0a53fa4cb62b56494f6554ec1a5e3b898a7dd"
# wandb.login(key="71f0a53fa4cb62b56494f6554ec1a5e3b898a7dd")

# %% [code] cell 3
# Check if the WANDB_API_KEY is set
if "WANDB_API_KEY" in os.environ:
    print("Logged in with API key.")
    user_info = wandb.api.viewer()
    print("Current user:", user_info["entity"], user_info["username"])
else:
    print("Not logged in.")

# %% [markdown]
# # Predicting Valid Human Wcd Environments

# %% [code] cell 5
# #f"simulated_valids_final{GRID_SIZE}.pkl"
# if GRID_SIZE == 6:
#     datasets = ["tmp_human_path-4.pt","tmp_human_path-3.pt","tmp_human_path-2.pt","tmp_human_path-1.pt","tmp_human_path-5.pt","tmp_human_grid6-6.pt"]
# else:
#     datasets = ["tmp_human_grid10-2.pt","tmp_human_grid10-0.pt","tmp_human_grid10-3.pt","tmp_human_grid10-4.pt","tmp_human_grid10-5.pt"]

# loaded_data = {}
# dataset=datasets[0]
# x_data = []
# y_data = []
# for dataset in datasets:
#     if not os.path.exists(f"data/grid{GRID_SIZE}/model_training/{dataset}"): continue
#     with open(f"data/grid{GRID_SIZE}/model_training/{dataset}", "rb") as f:
#         loaded_dataset = pickle.load(f)
#         print(dataset,len(loaded_dataset))
#         for i in range(loaded_dataset. __len__()):
#             x_i = loaded_dataset[i][0]
#             if x_i.shape[0]==5:
#                 x_i = x_i[0:4,:,:]

#             x_data.append(x_i.numpy())
#             y_data.append(loaded_dataset[i][1].unsqueeze(0).item())


# X_valid = np.stack(x_data)[:,0:4,:,:]
# # Y_valid = np.array(y_data)
# X_valid.shape

# %% [code] cell 6
#f"simulated_valids_final{GRID_SIZE}.pkl"
if GRID_SIZE == 6:
    datasets = [f"human_grid{GRID_SIZE}-100kall0.pt",f"human_grid{GRID_SIZE}-100kall1.pt",
                f"human_grid{GRID_SIZE}-100kall2.pt",f"human_grid{GRID_SIZE}-100kall3.pt",
               f"human_grid{GRID_SIZE}-100kall4.pt",f"human_grid{GRID_SIZE}-100kall5.pt"]
else:
    datasets = ["human_grid10-100kall0.pt","human_grid10-100kall1.pt","human_grid10-100kall2.pt","human_grid10-100kall3.pt",
               "human_grid10-100kall4.pt","human_grid10-100kall5.pt"]

loaded_data = {}
dataset=datasets[0]
x_data = []
y_data = []
for dataset in datasets:
    if not os.path.exists(f"data/grid{GRID_SIZE}/model_training/{dataset}"): continue
    with open(f"data/grid{GRID_SIZE}/model_training/{dataset}", "rb") as f:
        loaded_dataset = pickle.load(f)
        print(dataset,len(loaded_dataset))
        for i in range(loaded_dataset. __len__()):
            x_i = loaded_dataset[i][0]
            if x_i.shape[0]==5:
                x_i = x_i[0:4,:,:]

            x_data.append(x_i.numpy())
            y_data.append(loaded_dataset[i][1].unsqueeze(0).item())


X_invalid = np.stack(x_data)[:,0:4,:,:]
Y_invalid = (np.array(y_data)>0).astype(int)
X_invalid.shape

# %% [code] cell 7
Y_invalid.sum()/Y_invalid.shape[0]

# %% [code] cell 8
# y_invalid = -1 * np.ones(len(X_invalid))
# y_valid = 1 * np.ones(len(X_valid))

# # Combine data and labels
# X = np.concatenate((X_invalid, X_valid), axis=0)
# Y = np.concatenate((y_invalid, y_valid), axis=0)

# %% [code] cell 9
X = X_invalid
Y = Y_invalid

# %% [code] cell 11
x_train,x_test, y_train,y_test = train_test_split(X,Y, test_size=0.20)
x_train = np.concatenate([x_train[:, :, :, ::-1],x_train,x_train[:, :, ::-1, :],np.rot90(x_train, k=1, axes=(2, 3)),
                          np.rot90(x_train, k=3, axes=(2, 3)),np.rot90(x_train, k=2, axes=(2, 3)),
                          x_train.transpose(0, 1, 3, 2)])

y_train = np.concatenate([y_train,y_train,y_train,y_train,y_train,y_train,y_train])
x_train.shape, x_test.shape

# %% [code] cell 13
out_sample_set = CustomDataset(x_test,y_test)
with open(f"data/grid{GRID_SIZE}/model_training/dataset_{GRID_SIZE}.pkl", "wb") as f:
    pickle.dump(out_sample_set, f)

# %% [code] cell 14
x_test,x_val,y_test,y_val = train_test_split(x_test,y_test, test_size=0.95, random_state=50)
x_val.shape, x_test.shape

# %% [code] cell 16
x_test = torch.from_numpy(x_test).float().cuda()
y_test = torch.from_numpy(np.array(y_test)).float().cuda()
x_val = torch.from_numpy(x_val).float().cuda()
y_val = torch.from_numpy(np.array(y_val)).float().cuda()

# %% [code] cell 17
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

class CNN4(nn.Module):
    def __init__(self, n_channels=13,drop_out=0.01):
        super(CNN4, self).__init__()

        # Replace the first seven convolutional layers with ResNet50
        self.resnet50 = resnet18(pretrained=False)

        self.resnet50.conv1=self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1)
        self.resnet50.maxpool = nn.Identity()
        # self.resnet50.bn1 = nn.BatchNorm2d(8)

        # Keep the rest of the architecture as-is
        self.fc1 = nn.Linear(1000, 16)
        self.relu8 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(0.0001)

        self.fc2 = nn.Linear(16, 8)
        self.relu9 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(drop_out)

        self.fc3 = nn.Linear(8, 1)
        self.softplus = nn.functional.softplus
        self.sigmoid =torch.sigmoid

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

        x = self.sigmoid(self.fc3(x))

        return x

# %% [code] cell 18
def evaluate_and_log(best_model, x_val, y_val, epoch, wandb=None):
    def predict_in_batches(model, x, batch_size):
        n_batches = len(x) // batch_size + (len(x) % batch_size != 0)
        all_preds = []

        for i in range(n_batches):
            x_batch = x[i * batch_size: (i + 1) * batch_size]
            preds_batch = model(x_batch).detach()
            all_preds.append(preds_batch)

        return torch.cat(all_preds, dim=0)

    val_pred = (predict_in_batches(best_model, x_val, 1024)>0.5).float()  # Adjust batch size as per memory needs
    # print(val_pred==y_val)
    # if wandb:
    #     wandb.log({
    #         "epoch": epoch,
    #         "mse_val_small": nn.BCELoss()(val_pred[y_val <= GRID_SIZE/2], y_val.view(-1, 1)[y_val <= GRID_SIZE/2]).item(),
    #         "mse_val_big": nn.BCELoss()(val_pred[y_val > GRID_SIZE/2], y_val.view(-1, 1)[y_val > GRID_SIZE/2]).item(),
    #         "mse_val_loss": nn.BCELoss()(val_pred, y_val.view(-1, 1)).item(),
    #         "val_loss": nn.HuberLoss()(val_pred, y_val.view(-1, 1)).item(),
    #         "valid_mean_wcd": val_pred[y_val != INVALID_WCD].mean(),
    #         "invalid_mean_wcd": val_pred[y_val == INVALID_WCD].mean(),
    #         "valid_h_loss": nn.HuberLoss()(val_pred[y_val != INVALID_WCD], y_val.view(-1, 1)[y_val != INVALID_WCD]).item(),
    #         "invalid_h_loss": nn.HuberLoss()(val_pred[y_val == INVALID_WCD], y_val.view(-1, 1)[y_val == INVALID_WCD]).item(),
    #         "valid_mse_loss": nn.BCELoss()(val_pred[y_val != INVALID_WCD], y_val.view(-1, 1)[y_val != INVALID_WCD]).item(),
    #         "invalid_mse_loss": nn.BCELoss()(val_pred[y_val == INVALID_WCD], y_val.view(-1, 1)[y_val == INVALID_WCD]).item(),
    #     })

    accuracy = (val_pred==y_val).float().mean().item()
    print("Accuracy: ",accuracy)
    return nn.HuberLoss()(val_pred, y_val.view(-1, 1)).item() ,accuracy

# %% [code] cell 19
batch_size = 256 #512 for smaller (6,7,8)
dataset = TensorDataset(torch.from_numpy(x_train).float().cuda(), torch.from_numpy(np.array(y_train)).float().cuda())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# %% [code] cell 20
if GRID_SIZE in [6,7,8]:
    dropout = 0.01
    lambda_l2 =0.0 #0.1 regularization strength
    grad_clip =1e-3 # 1e-3 for 6,7,8
    lr=0.001 #0.005 for 6,7,8
    num_epochs = 12# 3 for 6,7,8
else:
    dropout = 0.0
    lambda_l2 =0.0 #0.1 regularization strength
    grad_clip =1e-3 # 1e-3 for 6,7,8
    lr=0.001 #0.005 for 6,7,8
    num_epochs = 12# 3 for 6,7,8

# %% [code] cell 21
# model = CustomCNN(n_channels=x_train.shape[1],drop_out=dropout,size = GRID_SIZE//2).cuda() #GRID_SIZE//2
random_seed=123
torch.manual_seed(seed=random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

model = CNN4(n_channels=x_train.shape[1],drop_out=dropout).cuda() #
# model = torch.load("models/wcd_nn_oracle_july6.pt")

# %% [code] cell 22
# # init_model = torch.load("models/wcd_10_init.pt")
# init_model = torch.load(f"../models/wcd_nn_model_{GRID_SIZE}_best.pt")

# model = init_model
# model.dropout1 = nn.Dropout(dropout)

# %% [code] cell 23
total_params = 0
for parameter in model.parameters():
    # print(parameter.shape)
    total_params += parameter.numel()  # numel() returns the total number of elements in the tensor

print(f"Total number of parameters: {total_params}") #636673

# %% [code] cell 24
# model = VGGNet(n_channels = x_train.shape[1]).cuda()

# %% [code] cell 25
use_wandb = False
if use_wandb:
    wandb.init(project='gridworld', save_code=False, config={"lambda_l2": lambda_l2, "grad_clip":grad_clip,
                                                              "n_train":x_train.shape[0],"GRID_SIZE":GRID_SIZE,"dropout":dropout,
                                                              "lr":lr,"batch_size":batch_size, "experiment":"gridworld"})
# create dataset and dataloader
# initialize model and optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[12,25,45,55], gamma=0.5)

# training loop

best_model = model
log_interval = 5

def train():
    training_loss = []
    val_mse_loss =[]
    val_huber_loss = []
    x_epochs = []

    lowest_loss = torch.inf
    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            # forward pass
            outputs = model(inputs)
            # targets = targets.cuda()

            # compute loss and perform backpropagation
            y_true = targets.view(-1, 1)
            loss = nn.BCELoss()(outputs, targets.view(-1, 1))
            mse_loss = loss.item()

            # l2_reg = lambda_l2 * torch.norm(torch.cat([p.view(-1) for p in model.parameters()]), p=2)  # L2 regularization term
            # loss += l2_reg

            loss.backward()
            utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if loss < lowest_loss:
                lowest_loss = loss.item()
                best_model = model

            if (i + 1) % 100*log_interval == 0:
                if use_wandb:
                    wandb.log({"loss": mse_loss})
                print(mse_loss)

        print(epoch,mse_loss)
        if (epoch + 1) % log_interval == 0:
            val_huber,val_mse = evaluate_and_log(best_model, x_val, y_val, epoch, wandb=wandb if use_wandb else None )
            val_mse_loss.append(val_mse)
            val_huber_loss.append(val_huber)
            training_loss.append(mse_loss)
            x_epochs.append(epoch)

            plt.plot(x_epochs,training_loss, label="Training")
            plt.plot(x_epochs,val_mse_loss, label="Validation")
            plt.show()

        scheduler.step()
        torch.save(best_model, f"models/valid_wcd_model_{GRID_SIZE}.pt")
    plt.plot(x_epochs,training_loss, label="Training")
    plt.plot(x_epochs,val_mse_loss, label="Validation")

    # Adding labels
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation MSE Loss")

    # Setting y-axis limits and grid lines in intervals of 5
    # plt.ylim(0, 30)
    # plt.yticks(range(0, 21, 5))
    plt.grid(axis='y')

    plt.legend()
    plt.show()
print('Starting Training')
# Start profiling
profiler = cProfile.Profile()
profiler.enable()

# Run the training loop
train()

# Stop profiling
profiler.disable()
# profiler.print_stats()

wandb.finish()

# %% [code] cell 27
nn.HuberLoss()(best_model(x_test),y_test.view(-1, 1))

# %% [code] cell 28
torch.mean(abs(best_model(x_test)-y_test.view(-1, 1)))

# %% [code] cell 29
torch.mean(abs(best_model(x_test)-y_test.view(-1, 1)))

# %% [code] cell 30
sns.kdeplot(best_model(x_test).cpu().detach().numpy(), fill=True,label="Pred")
sns.kdeplot(y_test.view(-1, 1).cpu().detach().numpy(), fill=True,label="True",color="green")
plt.legend()

# %% [code] cell 31
(best_model(x_test).cpu().detach().numpy()<0).sum()
(y_test<0).sum()

# %% [code] cell 32
torch.save(best_model,f"models/wcd_nn_oracle_{GRID_SIZE}.pt")

# %% [code] cell 33
x_test.shape
