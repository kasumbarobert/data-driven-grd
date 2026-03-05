import os
import pickle
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import argparse
import cProfile
from utils import *  # Assuming custom utils are available
import logging
import torch
import pandas as pd
import time
import random
import pickle
import numpy as np
from datetime import datetime
import gpytorch
from torch_geometric.data import Data,DataLoader

# Set the device (CUDA if available)
COMPUTE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def setup_experiment(model_dir, grid_size, log_file='model_performance.csv', summary_file='model_summary.csv'):
    """Sets up the experiment by creating the necessary directories and CSV files."""
    # Create directory for the model and logs
    os.makedirs(model_dir, exist_ok=True)

    # Initialize CSV file for model performance if it doesn't exist
    if not os.path.isfile(log_file):
        df = pd.DataFrame(columns=["Model", "Grid_Size", "Epochs", "Train_Loss", "Val_Loss", "MSE", "Huber_Loss"])
        df.to_csv(log_file, index=False)
    
    # Initialize CSV file for model summary if it doesn't exist
    if not os.path.isfile(summary_file):
        df = pd.DataFrame(columns=["Model", "Grid_Size", "Epochs", "Final_MSE", "Final_Huber_Loss", "Best_Model_Path"])
        df.to_csv(summary_file, index=False)

    return model_dir, log_file, summary_file

def load_data(grid_size, max_data_points = 1000):
    datasets = [f"gamma_1_grid{grid_size}_wcd.pt", f"simulated_valids_final{grid_size}_0.pkl", 
                f"simulated_valids_final{grid_size}_1.pkl", f"simulated_valids_fina{grid_size}.pkl"]
    x_data, y_data = [], []

    for dataset in datasets:
        file_path = f"data/grid{grid_size}/model_training/{dataset}"
        if not os.path.exists(file_path): continue
        with open(file_path, "rb") as f:
            loaded_dataset = pickle.load(f)
            for item in loaded_dataset:
                x_data.append(item[0].numpy())
                y_data.append(item[1].unsqueeze(0).item())
    
    X = np.stack(x_data)
    Y = np.array(y_data)

    logging.info(f"Total loaded data shapes, X {X.shape} is and Y is {Y.shape}")

    # Ensure that we don't sample more than the available data points
    num_samples = X.shape[0]
    sampled_points = min(max_data_points, num_samples)
    
    # Randomly sample indices
    random_indices = np.random.choice(num_samples, size=sampled_points, replace=False)

    # Truncate X and Y to the randomly sampled points
    X = X[random_indices]
    Y = Y[random_indices]

    return X, Y

def prepare_data(X, Y, augment = True):    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.05)
    # # Log data shapes
    logging.info(f"Training Data X Shape: {x_train.shape}, Validation Data X Shape: {x_test.shape}")
    logging.info(f"Training Data Y Shape: {y_train.shape}, Validation Data Y Shape: {y_test.shape}")

    if augment:
        x_train = np.concatenate([x_train[:, :, :, ::-1], x_train, x_train[:, :, ::-1, :], np.rot90(x_train, k=1, axes=(2, 3)),
                                np.rot90(x_train, k=3, axes=(2, 3)), np.rot90(x_train, k=2, axes=(2, 3)), x_train.transpose(0, 1, 3, 2)])
        y_train = np.concatenate([y_train] * 7)

        logging.info("Augmentation is applied!")

    # # Log data shapes
    logging.info(f"Training Data X Shape: {x_train.shape}, Validation Data X Shape: {x_test.shape}")
    logging.info(f"Training Data Y Shape: {y_train.shape}, Validation Data Y Shape: {y_test.shape}")
    
    return x_train, x_test, y_train, y_test


def plot_training_curves(training_loss, val_loss, epoch_numbers, model_dir, args):
    """Plots the training and validation loss curves and saves the plot."""
    # Create the figure and axis
    plt.figure(figsize=(12, 8), dpi=300)  # High resolution for publication quality

    # Plot the training and validation losses
    plt.plot(epoch_numbers, training_loss, label=f"Training Loss: {training_loss[-1]:.4f}", color='b', linestyle='-', linewidth=2)
    plt.plot(epoch_numbers, val_loss, label=f"Validation Loss: {val_loss[-1]:.4f}", color='r', linestyle='--', linewidth=2)

    # Add labels and title
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.title(f"{args.model_name} Training and Validation Loss Curves", fontsize=18)

    # Add a grid for better readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set font size for the legend
    plt.legend(fontsize=12, loc='best', frameon=False)

    # Add a text box with hyperparameters
    hyperparameters_text = '\n'.join([f"{param}: {value}" for param, value in vars(args).items() if isinstance(value, (int, float))])
    plt.gcf().text(0.15, 0.7, f"Hyperparameters:\n{hyperparameters_text}", fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.5'))

    # Save the figure to a file
    plt.tight_layout()  # Adjust layout for tight fitting
    plt.savefig(f"{model_dir}/training_curves.pdf", dpi=300, bbox_inches='tight')

    # Show the plot
    plt.close()

def train(model, train_dataloader, val_dataloader, optimizer, scheduler, criterion, num_epochs=40, log_file='model_performance.csv', args = None):
    """Train the model and log the results."""
    train_losses = []
    val_losses = []
    epoch_numbers = []

    lowest_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(COMPUTE_DEVICE), targets.to(COMPUTE_DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1, 1))
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        # Validate the model
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(COMPUTE_DEVICE), targets.to(COMPUTE_DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets.view(-1, 1))
                val_loss += loss.item()
            avg_val_loss = val_loss / len(val_dataloader)
            val_losses.append(avg_val_loss)

        # Log the results
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save the best model
        if avg_val_loss < lowest_loss:
            lowest_loss = avg_val_loss
            best_model = model
            torch.save(best_model, f"{args.model_log_dir}/{args.model_name}_model.pt")
        
        epoch_numbers.append(epoch)
        scheduler.step()

        # Save performance to CSV
        df = pd.read_csv(log_file)

    return best_model, train_losses, val_losses, epoch_numbers

def train_krr(model, train_dataloader, val_dataloader, criterion, log_file='model_performance.csv', args=None):
    """
    Train a Kernel Ridge Regression model and log the results.
    Args:
        model: The KernelRidgeRegression model to be trained.
        train_dataloader: DataLoader for training data.
        val_dataloader: DataLoader for validation data.
        criterion: Loss function (for logging purposes).
        log_file: File to log the performance metrics.
        args: Additional arguments for saving the model.
    Returns:
        The trained model and performance metrics.
    """
    import pandas as pd

    train_losses = []
    val_losses = []
    epoch_numbers = [0]  # KRR training is done in one step, but for logging consistency
    
    # Prepare training data
    train_inputs = []
    train_targets = []
    for inputs, targets in train_dataloader:
        train_inputs.append(inputs)
        train_targets.append(targets)

    train_inputs = torch.cat(train_inputs, dim=0)
    train_targets = torch.cat(train_targets, dim=0)

    # Train the KRR model
    logging.info("Training the Kernel Ridge Regression model...")
    model.fit(train_inputs, train_targets)

    # Calculate train loss
    train_outputs = model(train_inputs)
    train_loss = criterion(train_outputs, train_targets.view(-1, 1)).item()
    train_losses.append(train_loss)

    # Validate the model
    logging.info("Validating the model...")
    val_loss = 0.0
    val_inputs = []
    val_targets = []

    for inputs, targets in val_dataloader:
        val_inputs.append(inputs)
        val_targets.append(targets)

    val_inputs = torch.cat(val_inputs, dim=0)
    val_targets = torch.cat(val_targets, dim=0)

    val_outputs = model(val_inputs)
    val_loss = criterion(val_outputs, val_targets.view(-1, 1)).item()
    val_losses.append(val_loss)

    logging.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the trained model
    if args:
        torch.save(model, f"{args.model_log_dir}/{args.model_name}_model.pt")

    # Save performance to CSV
    performance_data = {
        'Epoch': epoch_numbers,
        'Train Loss': train_losses,
        'Val Loss': val_losses
    }
    df = pd.DataFrame(performance_data)
    df.to_csv(log_file, index=False)

    return model, train_losses, val_losses, epoch_numbers

def train_gp(model, likelihood, train_dataloader, val_dataloader, num_epochs=40, lr=0.1, log_file='gp_performance.csv', args=None):
    import pandas as pd

    # Extract the full training data
    for train_inputs, train_targets in train_dataloader:
        # Flatten inputs if necessary
        train_inputs = train_inputs.view(train_inputs.size(0), -1)  # Flatten everything except the batch dimension
        train_targets = train_targets.view(-1)  # Ensure targets are 1D
        break  # GPytorch requires all training data at once

    # Set the training data for the model
    model.set_train_data(inputs=train_inputs, targets=train_targets, strict=False)

    # Use the Adam optimizer
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Marginal Log Likelihood (for Gaussian Processes)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    train_losses = []
    val_losses = []
    epoch_numbers = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(train_inputs)  # Output should match train_targets size
        loss = -mll(output, train_targets)  # Ensure shapes match
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        epoch_numbers.append(epoch)

        # Validation
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            val_loss = 0.0
            for val_inputs, val_targets in val_dataloader:
                val_inputs = val_inputs.view(val_inputs.size(0), -1)  # Flatten validation inputs
                val_targets = val_targets.view(-1)  # Ensure targets are 1D
                val_output = model(val_inputs)
                val_pred = likelihood(val_output).mean
                val_loss += torch.nn.functional.mse_loss(val_pred, val_targets).item()
            val_loss /= len(val_dataloader)
            val_losses.append(val_loss)

        logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    # Save the model
    if args:
        torch.save(model.state_dict(), f"{args.model_log_dir}/{args.model_name}_gp_model.pt")

    # Save performance to CSV
    performance_data = {
        'Epoch': epoch_numbers,
        'Train Loss': train_losses,
        'Val Loss': val_losses
    }
    df = pd.DataFrame(performance_data)
    df.to_csv(log_file, index=False)

    return model, train_losses, val_losses, epoch_numbers






def train_gnn(model, train_dataloader, val_dataloader, optimizer, scheduler, criterion, num_epochs=40, log_file='model_performance.csv', args=None):
    """Train the GNN model and log the results."""
    
    # To keep track of losses and epochs
    train_losses = []
    val_losses = []
    epoch_numbers = []

    # To track the best model during training
    lowest_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Training loop
        for batch in train_dataloader:
            batch = batch.to(COMPUTE_DEVICE)  # Move batch to compute device
            
            optimizer.zero_grad()  # Zero out previous gradients
            outputs = model(batch)  # Forward pass (GNN model expects a 'Data' object)
            
            # Loss computation (assuming batch.y is the target label)
            loss = criterion(outputs, batch.y.view(-1, 1))  # Ensure targets are in the right shape
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update the model's parameters
            
            running_loss += loss.item()  # Accumulate loss

        # Calculate average training loss
        avg_train_loss = running_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Validate the model
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient tracking during evaluation
            val_loss = 0.0
            for batch in val_dataloader:
                batch = batch.to(COMPUTE_DEVICE)  # Move batch to compute device
                outputs = model(batch)  # Forward pass
                loss = criterion(outputs, batch.y.view(-1, 1))  # Loss computation
                val_loss += loss.item()  # Accumulate validation loss

            avg_val_loss = val_loss / len(val_dataloader)  # Average validation loss
            val_losses.append(avg_val_loss)

        # Log the results for the current epoch
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save the best model (based on validation loss)
        if avg_val_loss < lowest_loss:
            lowest_loss = avg_val_loss
            best_model = model
            torch.save(best_model.state_dict(), f"{args.model_log_dir}/{args.model_name}_model.pt")
        
        epoch_numbers.append(epoch)
        scheduler.step()  # Update the learning rate scheduler

        # Save performance to CSV
        df = pd.DataFrame({
            'epoch': epoch_numbers,
            'train_loss': train_losses,
            'val_loss': val_losses
        })
        df.to_csv(log_file, index=False)

    return best_model, train_losses, val_losses, epoch_numbers

def evaluate(model, x_out, y_out, model_dir, summary_file):
    """Evaluate the model on the output data."""
    model.eval()
    with torch.no_grad():
        y_out_preds = model(x_out)
        mse_loss = nn.MSELoss()(y_out_preds, y_out.view(-1, 1)).item()
        huber_loss = nn.HuberLoss()(y_out_preds, y_out.view(-1, 1)).item()
        
    logging.info(f"Final Evaluation - MSE Loss: {mse_loss:.4f}, Huber Loss: {huber_loss:.4f}")

    return mse_loss, huber_loss

def generate_model_name():
    nouns = [
        "Falcon", "Panther", "Hawk", "Tiger", "Eagle", "Phoenix", "Dragon", "Leopard", "Knight", "Viper",
        "Griffin", "Wolf", "Raven", "Cheetah", "Lion", "Ocelot", "Cobra", "Jaguar", "Serpent", "Bear",
        "Stallion", "Mustang", "Raptor", "Puma", "Tornado", "Cyclone", "Tsunami", "Thunder", "Blizzard", "Tempest",
        "Shark", "Orca", "Piranha", "Barracuda", "Manta", "Kraken", "Hydra", "Minotaur", "Centaur", "Pegasus",
        "Basilisk", "Chimera", "Sphinx", "Wyvern", "Cerberus", "Golem", "Yeti", "Mammoth", "Saber", "Scorpion",
        "Anaconda", "Komodo", "Gryphon", "Lynx", "Cougar", "Wolverine", "Badger", "Fox", "Hound", "Mongoose"
    ]
    
    verbs = [
        "Strike", "Run", "Fly", "Roar", "Charge", "Blaze", "Dash", "Glide", "Soar", "Dive",
        "Sprint", "Pounce", "Surge", "Flash", "Zoom", "Leap", "Rage", "Bolt", "Thrust", "Sweep",
        "Ascend", "Rush", "Storm", "Lunge", "Blast", "Thrive", "Ambush", "Pierce", "Unleash", "Ignite",
        "Climb", "Swoop", "Gallop", "Hurdle", "Skim", "Skate", "Slide", "Trek", "Vault", "Whirl",
        "Whiz", "Whip", "Whisk", "Whirlwind", "Zigzag", "Zoom", "Barrel", "Bound", "Burst", "Cannon",
        "Catapult", "Charge", "Chase", "Cruise", "Dart", "Dash", "Flee", "Gallop", "Hasten", "Hurtle",
        "Jet", "Lunge", "Plunge", "Race", "Rocket", "Rush", "Scamper", "Scurry", "Shoot", "Skim"
    ]
    
    # Generate a 4-digit time-based number (e.g., last 4 digits of current time in seconds)
    timestamp = int(time.time())  # Current time in seconds
    number = timestamp % 10000  # Get the last 4 digits
    # Create a new random generator instance
    local_random = random.Random()

    # Use the local_random instance to make choices
    noun = local_random.choice(nouns)
    verb = local_random.choice(verbs)

    model_name = f"{noun}-{verb}-{number:04d}".lower()  # Ensure the number is always 4 digits
    return model_name


# Function to log parameters and results to CSV
def log_parameters_and_results(args, summary_file, train_mse, val_mse):
    # Capture current date and time as the first column
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Convert argparse arguments to a dictionary
    args_dict = vars(args)
    
    # Add additional model evaluation results and timestamp to the dictionary
    args_dict["date_time"] = current_datetime
    args_dict["model_file"] = f"{args.model_name}_model.pt"
    args_dict["train_mse"] = train_mse
    args_dict["val_mse"] = val_mse
    
    # Check if the CSV file exists
    if os.path.exists(summary_file):
        # Load existing DataFrame
        df = pd.read_csv(summary_file)
        
        # Ensure no duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
    else:
        # Create a new DataFrame with columns derived from the current args_dict
        df = pd.DataFrame(columns=list(args_dict.keys()))
    
    # Ensure all columns in the new row are in the existing DataFrame
    for col in args_dict.keys():
        if col not in df.columns:
            df[col] = None  # Add missing columns
    
    # Align new_row with existing DataFrame columns
    new_row = pd.DataFrame([args_dict])[df.columns]
    
    # Use pd.concat to append the new row
    df = pd.concat([df, new_row], ignore_index=True)
    
    # Ensure date_time is the first column
    column_order = ["date_time"] + [col for col in df.columns if col != "date_time"]
    df = df[column_order]
    
    # Save the updated DataFrame to the CSV file
    df.to_csv(summary_file, index=False)

def main(args):
    """
    Main function for training and evaluating a model (CNN or Kernel Ridge Regression).
    Handles data preparation, model initialization, training, and result logging.
    """
    # Setup experiment directory and log files
    model_dir, log_file, summary_file = setup_experiment(args.model_log_dir, args.grid_size)

    # Load and prepare data
    X, Y = load_data(args.grid_size, max_data_points=args.max_data_points)
    x_train, x_test, y_train, y_test = prepare_data(X, Y, augment = True if args.model_type in ("cnn","gnn") else False)

    # Create datasets and dataloaders
    batch_size = args.batch_size
    train_dataset = TensorDataset(
        torch.from_numpy(x_train).float(), 
        torch.from_numpy(np.array(y_train)).float()
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(
        torch.from_numpy(x_test).float(), 
        torch.from_numpy(np.array(y_test)).float()
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Define the loss criterion
    criterion = nn.MSELoss()
    # 

    # Initialize model, optimizer, and scheduler based on model type
    if args.model_type == "cnn":
        model = CNN4(n_channels=x_train.shape[1], drop_out=0.01).to(COMPUTE_DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[12, 25, 45, 55], gamma=0.5
        )

        # Train the model
        best_model, train_losses, val_losses, epoch_numbers = train(
            model, train_dataloader, val_dataloader, optimizer, scheduler, 
            criterion, num_epochs=args.epochs, log_file=log_file, args=args
        )
    elif args.model_type == "transformer":
        patch_size = 4
        model = VisionTransformer(n_channels=x_train.shape[1],height=x_train.shape[2], width=x_train.shape[3],patch_size =patch_size).to(COMPUTE_DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[12, 25, 45, 55], gamma=0.5
        )

        # Train the model
        best_model, train_losses, val_losses, epoch_numbers = train(
            model, train_dataloader, val_dataloader, optimizer, scheduler, 
            criterion, num_epochs=args.epochs, log_file=log_file, args=args
        )
    elif args.model_type == "linear":
        model = LinearModel(n_channels=x_train.shape[1],height=x_train.shape[2], width=x_train.shape[3]).to(COMPUTE_DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[12, 25, 45, 55], gamma=0.5
        )

        # Train the model
        best_model, train_losses, val_losses, epoch_numbers = train(
            model, train_dataloader, val_dataloader, optimizer, scheduler, 
            criterion, num_epochs=args.epochs, log_file=log_file, args=args
        )
    elif args.model_type == "gnn":
        model = GNNModel(n_channels=x_train.shape[1], drop_out=0.01).to(COMPUTE_DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        train_dataset = create_graph_dataset(
            torch.from_numpy(x_train).float(), 
            torch.from_numpy(np.array(y_train)).float()
        )
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = create_graph_dataset(
            torch.from_numpy(x_test).float(), 
            torch.from_numpy(np.array(y_test)).float()
        )
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[12, 25, 45, 55], gamma=0.5
        )

        # Train the model
        best_model, train_losses, val_losses, epoch_numbers = train_gnn(
            model, train_dataloader, val_dataloader, optimizer, scheduler, 
            criterion, num_epochs=args.epochs, log_file=log_file, args=args
        )

    elif args.model_type == "krr":
        model = KernelRidgeRegression(
            input_dim=x_train.shape[1] * x_train.shape[2] * x_train.shape[3], 
            alpha=0.1, kernel="rbf", gamma=0.01
        )

        # Train the Kernel Ridge Regression model
        best_model, train_losses, val_losses, epoch_numbers = train_krr(
            model, train_dataloader, val_dataloader, criterion, 
            log_file=f"{args.model_log_dir}/krr_performance.csv", args=args
        )
    elif args.model_type == "gp":
        # Prepare the full training data
        train_inputs, train_targets = next(iter(train_dataloader))
        train_inputs = train_inputs.view(-1, train_inputs.shape[-1])
        train_targets = train_targets.view(-1)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GaussianProcessRegressionModel(train_x=train_inputs, train_y=train_targets, likelihood=likelihood)

        trained_model, train_losses, val_losses, epoch_numbers = train_gp(
            model, likelihood, train_dataloader, val_dataloader,
            num_epochs=args.epochs, lr=args.lr, log_file=f"{args.model_log_dir}/gp_performance.csv", args=args
        )

    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    # Plot training curves
    plot_training_curves(train_losses, val_losses, epoch_numbers, model_dir, args)

    # Log parameters and results
    log_parameters_and_results(
        args, 
        summary_file=f"{args.summary_file_dir}/performance_summary.csv", 
        train_mse=train_losses[-1], 
        val_mse=val_losses[-1]
    )

    print("Model training completed model name: ", args.model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a model.")
    
    # Arguments
    # parser.add_argument('--model_name', type=str, required=True, help='The name of the model (folder to save results).')
    parser.add_argument('--grid_size', type=int, default = 13, help='Grid size for the model.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training.')
    parser.add_argument('--max_data_points', type=int, default=1000, help='Maximum amount of data in model training')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for the optimizer.')
    parser.add_argument("--run_mode", type=str, default="debug", choices=["debug", "run"], help="Script  run mode")
    parser.add_argument("--model_type", type=str, default="cnn", choices=["krr", "cnn","gp","linear","gnn","transformer"], help="Model architecture")

    args = parser.parse_args()
    args.compute_device = COMPUTE_DEVICE

    args.model_name = generate_model_name()

    args.model_log_dir = f"./models/wcd_prediction/grid{args.grid_size}/"
    
    os.makedirs(args.model_log_dir, exist_ok=True)

    if args.run_mode == "debug":
        args.model_log_dir = os.path.join(args.model_log_dir,  "debug_logs")
        args.model_name = f"debug-{args.model_name}"
        args.max_data_points = 200
        args.epochs = 3
    else:
        args.model_log_dir = os.path.join(args.model_log_dir,  "training_logs")

    args.summary_file_dir = args.model_log_dir
    args.model_log_dir = os.path.join(args.model_log_dir,   args.model_name)
    os.makedirs(args.model_log_dir, exist_ok = True)

     # Setup logging
    log_file = os.path.join(args.model_log_dir, "training.logs")

    # Set up logging
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Log the arguments
    logging.info("Training configuration:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")

    # Call the main function with parsed arguments
    main(args)
