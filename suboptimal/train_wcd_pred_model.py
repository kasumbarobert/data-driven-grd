import os
import pickle
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import argparse
import cProfile
from utils_suboptimal import *  # Import suboptimal utilities
import logging
import torch
import pandas as pd
import time
import random
import pickle
import numpy as np
from datetime import datetime

# Set the device (CUDA if available)
COMPUTE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class NpyDataset(Dataset):
    def __init__(self, x_file_path, y_file_path, chunk_size=2048):
        self.x_file_path = x_file_path
        self.y_file_path = y_file_path
        self.chunk_size = chunk_size
        
        # Get file size to determine number of samples
        with open(self.x_file_path, 'rb') as f:
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()
        
        # Estimate number of samples (approximate)
        self.num_samples = file_size // (chunk_size * 4 * 4 * 4 * 4)  # 4 channels, 4x4 grid
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Load data from file
        with open(self.x_file_path, 'rb') as f:
            f.seek(idx * self.chunk_size * 4 * 4 * 4 * 4 * 4)  # Seek to position
            x_data = np.load(f)
        
        with open(self.y_file_path, 'rb') as f:
            f.seek(idx * self.chunk_size * 4)  # Seek to position
            y_data = np.load(f)
        
        return torch.tensor(x_data[idx % self.chunk_size], dtype=torch.float32), torch.tensor(y_data[idx % self.chunk_size], dtype=torch.float32)

def augment_data(chunk):
    """Augment data with various transformations."""
    augmented_chunk = np.concatenate([
        chunk,
        chunk[:, :, :, ::-1],        # Flip along the last axis
        chunk[:, :, ::-1, :],        # Flip along the second to last axis
        np.rot90(chunk, k=1, axes=(2, 3)), # Rotate 90 degrees
        np.rot90(chunk, k=3, axes=(2, 3)), # Rotate 270 degrees
        np.rot90(chunk, k=2, axes=(2, 3)), # Rotate 180 degrees
        chunk.transpose(0, 1, 3, 2)  # Transpose axes
    ])
    return augmented_chunk

def load_data(grid_size, K, max_data_points=1000):
    """Load training data for suboptimal setting."""
    if grid_size == 10:
        datasets = [
            f"archive-data/hyperbol_simulated_envs_K{K}_1.pkl",
            f"archive-data/hyperbol_simulated_envs_K{K}_2.pkl",
            f"archive-data/hyperbol_simulated_envs_K{K}_3.pkl",
            f"archive-data/hyperbol_simulated_envs_K{K}_4.pkl",
            f"archive-data/hyperbol_simulated_envs_K{K}_5.pkl",
            f"archive-data/hyperbol_simulated_envs_K{K}_6.pkl",
            f"archive-data/hyperbol_simulated_envs_K{K}_7.pkl",
            f"archive-data/hyperbol_simulated_envs_K{K}_8.pkl",
            f"hyperbol_simulated_envs_K{K}_0.pkl",
            f"simulated_valids_final10_K{K}.pkl"
        ]
    elif grid_size == 6:
        datasets = [
            f"hyperbol_simulated_envs_K{K}_0.pkl",
            f"hyperbol_simulated_envs_K{K}_1.pkl",
            f"hyperbol_simulated_envs_K{K}_2.pkl",
            f"hyperbol_simulated_envs_K{K}_3.pkl",
            f"simulated_valids_final{grid_size}_ALL_MODS_K{K}.pkl",
            f"simulated_valids_final{grid_size}_BLOCKING_ONLY_K{K}.pkl",
            f"simulated_valids_final{grid_size}_BOTH_UNIFORM_K{K}.pkl",
            f"simulated_valids_final{grid_size}_K{K}.pkl"
        ]
    else:
        datasets = [f"hyperbol_simulated_envs_K{K}_0.pkl"]
    
    x_data = []
    y_data = []
    
    for dataset in datasets:
        file_path = f"data/grid{grid_size}/model_training/{dataset}"
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, "rb") as f:
            loaded_dataset = pickle.load(f)
            print(f"Loading {dataset}: {len(loaded_dataset)} samples")
            
            for i in range(loaded_dataset.__len__()):
                x_i = loaded_dataset[i][0]
                if x_i.shape[0] == 5:
                    x_i = x_i[0:4, :, :]
                    
                x_data.append(x_i.numpy())
                y_data.append(loaded_dataset[i][1].unsqueeze(0).item())
    
    X = np.stack(x_data)[:, 0:4, :, :]
    Y = np.array(y_data)
    
    logging.info(f"Total loaded data shapes, X {X.shape} and Y is {Y.shape}")
    
    # Ensure that we don't sample more than the available data points
    num_samples = X.shape[0]
    sampled_points = min(max_data_points, num_samples)
    
    # Randomly sample indices
    random_indices = np.random.choice(num_samples, size=sampled_points, replace=False)
    
    # Truncate X and Y to the randomly sampled points
    X = X[random_indices]
    Y = Y[random_indices]
    
    return X, Y

def prepare_data(X, Y, augment=True):
    """Prepare and split data for training."""
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.10)
    
    logging.info(f"Training Data X Shape: {x_train.shape}, Validation Data X Shape: {x_test.shape}")
    logging.info(f"Training Data Y Shape: {y_train.shape}, Validation Data Y Shape: {y_test.shape}")
    
    if augment:
        x_train = augment_data(x_train)
        y_train = np.concatenate([y_train] * 7)
        logging.info("Augmentation is applied!")
    
    logging.info(f"After augmentation - Training Data X Shape: {x_train.shape}, Validation Data X Shape: {x_test.shape}")
    logging.info(f"After augmentation - Training Data Y Shape: {y_train.shape}, Validation Data Y Shape: {y_test.shape}")
    
    return x_train, x_test, y_train, y_test

def evaluate_and_log(model, x_val, y_val, epoch, wandb=None):
    """Evaluate model and log metrics."""
    model.eval()
    with torch.no_grad():
        val_pred = model(x_val.cuda())
        val_mse = nn.MSELoss()(val_pred, y_val.view(-1, 1).cuda()).item()
        val_huber = nn.HuberLoss()(val_pred, y_val.view(-1, 1).cuda()).item()
        
        if wandb:
            wandb.log({"val_mse": val_mse, "val_huber": val_huber, "epoch": epoch})
    
    model.train()
    return val_huber, val_mse

def plot_training_curves(training_loss, val_loss, epoch_numbers, model_dir, args):
    """Plots the training and validation loss curves and saves the plot."""
    plt.figure(figsize=(12, 8), dpi=300)
    
    plt.plot(epoch_numbers, training_loss, label=f"Training Loss: {training_loss[-1]:.4f}", color='b', linestyle='-', linewidth=2)
    plt.plot(epoch_numbers, val_loss, label=f"Validation Loss: {val_loss[-1]:.4f}", color='r', linestyle='--', linewidth=2)
    
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.title(f"Suboptimal WCD Prediction Training and Validation Loss Curves", fontsize=18)
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12, loc='best', frameon=False)
    
    # Add hyperparameters text
    hyperparameters_text = '\n'.join([f"{param}: {value}" for param, value in vars(args).items() if isinstance(value, (int, float))])
    plt.gcf().text(0.15, 0.7, f"Hyperparameters:\n{hyperparameters_text}", fontsize=12, 
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.savefig(f"{model_dir}/training_curves.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def train(model, train_dataloader, val_dataloader, optimizer, scheduler, criterion, num_epochs=40, log_file='model_performance.csv', args=None):
    """Train the model and log the results."""
    train_losses = []
    val_losses = []
    epoch_numbers = []
    
    lowest_loss = float('inf')
    best_model = model
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, targets) in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            outputs = model(inputs.to(COMPUTE_DEVICE))
            y_true = targets.view(-1, 1).to(COMPUTE_DEVICE)
            loss = criterion(outputs, y_true)
            mse_loss = loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i + 1) % 1000 == 0:
                logging.info(f"Epoch {epoch+1}, Batch {i+1}, Loss: {mse_loss:.4f}")
        
        avg_train_loss = running_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        # Validate the model
        if (epoch + 1) % 2 == 0:  # Validate every 2 epochs
            val_huber, val_mse = evaluate_and_log(model, args.x_val, args.y_val, epoch)
            val_losses.append(val_mse)
            epoch_numbers.append(epoch)
            
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_mse:.4f}")
            
            # Save the best model
            if val_mse < lowest_loss:
                lowest_loss = val_mse
                best_model = model
                torch.save(best_model, f"{args.model_log_dir}/{args.model_name}_model.pt")
        
        scheduler.step()
    
    return best_model, train_losses, val_losses, epoch_numbers

def setup_experiment(model_dir, grid_size, log_file='model_performance.csv', summary_file='model_summary.csv'):
    """Sets up the experiment by creating the necessary directories and CSV files."""
    os.makedirs(model_dir, exist_ok=True)
    
    if not os.path.isfile(log_file):
        df = pd.DataFrame(columns=["Model", "Grid_Size", "Epochs", "Train_Loss", "Val_Loss", "MSE", "Huber_Loss"])
        df.to_csv(log_file, index=False)
    
    if not os.path.isfile(summary_file):
        df = pd.DataFrame(columns=["Model", "Grid_Size", "Epochs", "Final_MSE", "Final_Huber_Loss", "Best_Model_Path"])
        df.to_csv(summary_file, index=False)
    
    return model_dir, log_file, summary_file

def generate_model_name():
    """Generate a unique model name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"suboptimal_wcd_model_{timestamp}"

def log_parameters_and_results(args, summary_file, train_mse, val_mse):
    """Log parameters and results to CSV file."""
    args_dict = {
        "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": args.model_name,
        "grid_size": args.grid_size,
        "K": args.K,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "dropout": args.dropout,
        "grad_clip": args.grad_clip,
        "final_train_mse": train_mse,
        "final_val_mse": val_mse,
        "best_model_path": f"{args.model_log_dir}/{args.model_name}_model.pt"
    }
    
    df = pd.read_csv(summary_file)
    
    for col in args_dict.keys():
        if col not in df.columns:
            df[col] = None
    
    new_row = pd.DataFrame([args_dict])[df.columns]
    df = pd.concat([df, new_row], ignore_index=True)
    
    column_order = ["date_time"] + [col for col in df.columns if col != "date_time"]
    df = df[column_order]
    
    df.to_csv(summary_file, index=False)

def main(args):
    """Main function for training the suboptimal WCD prediction model."""
    # Setup experiment directory and log files
    model_dir, log_file, summary_file = setup_experiment(args.model_log_dir, args.grid_size)
    
    # Load and prepare data
    X, Y = load_data(args.grid_size, args.K, max_data_points=args.max_data_points)
    x_train, x_test, y_train, y_test = prepare_data(X, Y, augment=True)
    
    # Store validation data for evaluation
    args.x_val = torch.from_numpy(x_test).float()
    args.y_val = torch.from_numpy(y_test).float()
    
    # Create datasets and dataloaders
    batch_size = args.batch_size
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_train).float(), 
        torch.from_numpy(np.array(y_train)).float()
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_test).float(), 
        torch.from_numpy(np.array(y_test)).float()
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    # Define the loss criterion
    criterion = nn.MSELoss()
    
    # Initialize model
    model = CNN4(n_channels=x_train.shape[1], drop_out=args.dropout).to(COMPUTE_DEVICE)
    
    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[12, 25, 45, 55], gamma=0.5
    )
    
    # Train the model
    best_model, train_losses, val_losses, epoch_numbers = train(
        model, train_dataloader, val_dataloader, optimizer, scheduler, 
        criterion, num_epochs=args.epochs, log_file=log_file, args=args
    )
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, epoch_numbers, model_dir, args)
    
    # Log parameters and results
    log_parameters_and_results(
        args, 
        summary_file=f"{args.summary_file_dir}/performance_summary.csv", 
        train_mse=train_losses[-1], 
        val_mse=val_losses[-1]
    )
    
    print("Model training completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train suboptimal WCD prediction model.")
    
    # Arguments
    parser.add_argument('--grid_size', type=int, default=6, help='Grid size for the model.')
    parser.add_argument('--K', type=int, default=4, help='Hyperbolic discounting parameter.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training.')
    parser.add_argument('--max_data_points', type=int, default=1000, help='Maximum amount of data in model training')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument("--run_mode", type=str, default="debug", choices=["debug", "run"], help="Script run mode")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--grad_clip", type=float, default=1e-3, help="Gradient clipping value")
    
    args = parser.parse_args()
    args.compute_device = COMPUTE_DEVICE
    
    args.model_name = generate_model_name()
    args.model_log_dir = f"./models/wcd_prediction/grid{args.grid_size}/"
    
    os.makedirs(args.model_log_dir, exist_ok=True)
    
    if args.run_mode == "debug":
        args.model_log_dir = os.path.join(args.model_log_dir, "debug_logs")
        args.model_name = f"debug-{args.model_name}"
        args.max_data_points = 200
        args.epochs = 3
    else:
        args.model_log_dir = os.path.join(args.model_log_dir, "training_logs")
    
    args.summary_file_dir = args.model_log_dir
    args.model_log_dir = os.path.join(args.model_log_dir, args.model_name)
    os.makedirs(args.model_log_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(args.model_log_dir, "training.logs")
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Log the arguments
    logging.info("Training configuration:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")
    
    # Call the main function with parsed arguments
    main(args) 