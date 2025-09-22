# Gridworld MDP Optimization Framework - Suboptimal Agents

This folder contains the implementation of a machine learning-based optimization framework for Gridworld Markov Decision Processes (MDPs) assuming **Suboptimal Agents** with hyperbolic discounting. The framework focuses on optimizing Worst-Case Distance (WCD) values through environment modifications using our proposed approach. The key idea is to train a model to predict WCD for suboptimal agents and then use the trained model to modify the environment to minimize WCD.

## Overview

The framework consists of several key components:
- **MDP Implementation**: Core Gridworld MDP functionality with hyperbolic discounting (`mdp.py`)
- **Utility Functions**: Helper functions and model architectures for suboptimal agents (`utils_suboptimal.py`)
- **Data Generation**: Training data generation for suboptimal agents (`simulate_data_generation.py`)
- **Model Training**: WCD prediction model training (`train_wcd_pred_model.py`)
- **Optimization**: Main optimization experiments (`run_optimization_subopt.py`)
- **Baselines**: Comparison baseline experiments (`run_baseline_experiments_subopt.py`)

## Key Differences from Optimal Setting

- **Suboptimal Agents**: Uses hyperbolic discounting with parameter K instead of optimal agents
- **Simplified Model Architecture**: Uses CNN4 with ResNet18 backbone
- **Hyperbolic Q-Function**: Implements `computeHyperbolicQ` for suboptimal agent behavior
- **Data Generation**: Includes script to generate training data for suboptimal scenarios

## Directory Structure

```
suboptimal/
├── mdp.py                           # Core MDP implementation with hyperbolic discounting
├── utils_suboptimal.py              # Utility functions for suboptimal agents
├── simulate_data_generation.py      # Training data generation
├── train_wcd_pred_model.py          # WCD prediction model training
├── run_optimization_subopt.py       # Main optimization experiments
├── run_baseline_experiments_subopt.py # Baseline experiments
├── run_exp_script_subopt.sh         # Shell script for batch execution
├── baselines/                       # Baseline experiment implementations
├── data/                           # Training and test data
├── models/                         # Trained models
├── plot_data/                      # Analysis data
└── plots/                          # Generated plots and visualizations
```

## Quick Start Guide

### Prerequisites

1. **Data**: Generate training data using `simulate_data_generation.py` or ensure data exists in `data/grid{size}/model_training/`
2. **Dependencies**: Install required packages (PyTorch, NumPy, Matplotlib, etc.)
3. **GPU**: Recommended for faster training and optimization

### Step-by-Step Execution

#### 1. Generate Training Data (if needed)

```bash
python simulate_data_generation.py --grid_size 6 --K 4
```

**Parameters:**
- `--grid_size`: Size of the gridworld (default: 10)
- `--K`: Hyperbolic discounting parameter (default: 4)

**Output:** Training data saved in `./data/grid{size}/model_training/hyperbol_simulated_envs_K{K}_0.pkl`

#### 2. Train WCD Prediction Model

```bash
python train_wcd_pred_model.py --grid_size 6 --K 4 --epochs 20 --batch_size 512 --lr 0.001
```

**Parameters:**
- `--grid_size`: Size of the gridworld (default: 6)
- `--K`: Hyperbolic discounting parameter (default: 4)
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Training batch size (default: 512)
- `--lr`: Learning rate (default: 0.001)
- `--dropout`: Dropout rate (default: 0.1)
- `--grad_clip`: Gradient clipping value (default: 1e-3)
- `--max_data_points`: Maximum data points to use (default: 1000)
- `--run_mode`: Script mode - "debug" or "run" (default: "debug")

**Output:** Trained model saved in `./models/wcd_prediction/grid{size}/training_logs/{model_id}/`

#### 3. Run Optimization Experiments

```bash
python run_optimization_subopt.py --grid_size 6 --experiment_type ALL_MODS --ratio 1_1 --wcd_pred_model_id {model_id} --num_instances 5 --max_iter 20 --K 4
```

**Parameters:**
- `--grid_size`: Grid size (default: 6)
- `--experiment_type`: Type of experiment (BLOCKING_ONLY, ALL_MODS, BOTH_UNIFORM)
- `--ratio`: Ratio for modifications (e.g., "1_1", "3_1", "1_3")
- `--wcd_pred_model_id`: ID of the trained model from step 2
- `--num_instances`: Number of environment instances to process
- `--max_iter`: Maximum optimization iterations
- `--K`: Hyperbolic discounting parameter

**Output:** Results saved in `./wcd_optim_results/ml-our-approach/grid{size}/{model_id}/`

#### 4. Run Baseline Experiments (Optional)

```bash
python run_baseline_experiments_subopt.py --grid_size 6 --experiment_type ALL_MODS --ratio 1_1 --K 4
```

**Output:** Baseline results saved in `./baselines/data/grid{size}/`

## Experiment Types

### 1. BLOCKING_ONLY
- Only allows blocking actions (adding obstacles)
- Focuses on strategic placement of barriers for suboptimal agents

### 2. ALL_MODS
- Allows both blocking and unblocking actions in a ratio (e.g., 1_1 means 1:1)
- More flexible optimization with ratio constraints

### 3. BOTH_UNIFORM
- Uniform cost for both blocking and unblocking
- Balanced approach to environment modification

## Hyperbolic Discounting Parameter (K)

The K parameter controls the degree of suboptimality in agent behavior:
- **Lower K values**: More suboptimal agents (shorter planning horizons)
- **Higher K values**: More optimal agents (longer planning horizons)
- **Typical range**: 0.1 to 10.0

## Model Architecture

The suboptimal framework uses a simplified model architecture:
- **CNN4**: Convolutional Neural Network with ResNet18 backbone
- Optimized for suboptimal agent WCD prediction
- Single model type (no transformer, GNN, etc.)

## Key Features

### Optimization Algorithm
- Gradient-based optimization using trained WCD prediction models
- Constraint handling for valid environment modifications
- Multiple experiment types with different modification strategies
- Hyperbolic discounting for suboptimal agent simulation

### Analysis Capabilities
- Comprehensive performance comparison between approaches
- Statistical analysis with standard errors
- Visualization of WCD reduction vs. modification budget
- Time complexity analysis for suboptimal scenarios

### Scalability
- Batch processing for multiple environment instances
- GPU acceleration for model training and optimization
- Configurable parameters for different grid sizes and K values

## Output Files

### Data Generation Outputs
- `data/grid{size}/model_training/hyperbol_simulated_envs_K{K}_0.pkl`: Generated training data

### Training Outputs
- `models/wcd_prediction/grid{size}/training_logs/{model_id}/`: Trained models and training logs
- `training_curves.pdf`: Training and validation loss curves
- `training.logs`: Detailed training logs
- `performance_summary.csv`: Model performance summary

### Optimization Outputs
- `wcd_optim_results/ml-our-approach/grid{size}/{model_id}/`: Optimization results
- JSON files with lambda pairs and performance metrics
- Environment modification data

## Troubleshooting

### Common Issues

1. **Missing Data**: Generate training data using `simulate_data_generation.py`
2. **GPU Memory**: Reduce batch size or use CPU if GPU memory is insufficient
3. **Model Not Found**: Verify the `wcd_pred_model_id` matches a trained model
4. **K Parameter**: Ensure K parameter is consistent across training and optimization

### Debug Mode

Use debug mode for testing with smaller datasets:
```bash
python train_wcd_pred_model.py --run_mode debug
```

## Advanced Usage

### Custom Data Generation
```bash
python simulate_data_generation.py --grid_size 8 --K 2
```

### Different K Values
```bash
python train_wcd_pred_model.py --K 8 --grid_size 6 --epochs 50
```

### Custom Training Parameters
```bash
python train_wcd_pred_model.py --grid_size 10 --K 4 --epochs 100 --lr 0.0005 --dropout 0.5 --grad_clip 1e-4
```

### Batch Processing
Use the shell script for batch processing:
```bash
bash run_exp_script_subopt.sh
```

## Dependencies

Ensure to install the packages in `../environment.yml`. We encourage using conda to setup the environment.

