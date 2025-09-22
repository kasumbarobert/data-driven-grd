# Gridworld MDP Optimization Framework

This folder contains the implementation of a machine learning-based optimization framework for Gridworld Markov Decision Processes (MDPs) assuming Optimal Agents. The framework focuses on optimizing Worst-Case Distance (WCD) values through environment modifications using our proposed approach. The key idea is to train a model to predict WCD and the use the trained model to modify the environment to minimize WCD.

## Overview

The framework consists of several key components:
- **MDP Implementation**: Core Gridworld MDP functionality (`mdp.py`)
- **Utility Functions**: Helper functions and model architectures (`utils.py`)
- **Model Training**: WCD prediction model training (`train_wcd_pred_model.py`)
- **Optimization**: Main optimization experiments (`run_optimization_opt.py`)
- **Baselines**: Comparison baseline experiments (`baselines/`)
- **Analysis**: Data processing and visualization (`prepare_data_for_analysis.py`, `analyse_and_plot.py`)

## Directory Structure

```
optimal/
├── mdp.py                           # Core MDP implementation
├── utils.py                         # Utility functions and model architectures
├── train_wcd_pred_model.py          # WCD prediction model training
├── run_optimization_opt.py          # Main optimization experiments
├── prepare_data_for_analysis.py     # Data processing for analysis
├── analyse_and_plot.py              # Visualization and final analysis
├── baselines/                       # Baseline experiment implementations
├── scripts/                         # Shell scripts for batch execution
├── data/                           # Training and test data
├── models/                         # Trained models
├── wcd_optim_results/              # Optimization experiment results
├── summary_data/                   # Processed analysis data
└── plots/                          # Generated plots and visualizations
```

## Quick Start Guide

### Prerequisites

1. **Data**: Ensure training data is available in `data/grid{size}/model_training/`
2. **Dependencies**: Install required packages (PyTorch, NumPy, Matplotlib, etc.)
3. **GPU**: Recommended for faster training and optimization

### Step-by-Step Execution

#### 1. Train WCD Prediction Model

```bash
python train_wcd_pred_model.py --grid_size 13 --model_type cnn --epochs 100 --batch_size 512 --lr 0.01
```

**Parameters:**
- `--grid_size`: Size of the gridworld (default: 13)
- `--model_type`: Model architecture (cnn, transformer, linear, gnn, krr, gp)
- `--epochs`: Number of training epochs
- `--batch_size`: Training batch size
- `--lr`: Learning rate

**Output:** Trained model saved in `./models/wcd_prediction/grid{size}/training_logs/{model_id}/`

#### 2. Run Optimization Experiments

```bash
python run_optimization_opt.py --grid_size 13 --experiment_type ALL_MODS --ratio 1_1 --wcd_pred_model_id {model_id} --num_instances 5 --max_iter 20
```

**Parameters:**
- `--grid_size`: Grid size (default: 10)
- `--experiment_type`: Type of experiment (BLOCKING_ONLY, ALL_MODS, BOTH_UNIFORM)
- `--ratio`: Ratio for modifications (e.g., "1_1", "3_1", "1_3")
- `--wcd_pred_model_id`: ID of the trained model from step 1
- `--num_instances`: Number of environment instances to process
- `--max_iter`: Maximum optimization iterations

**Output:** Results saved in `./wcd_optim_results/ml-our-approach/grid{size}/{model_id}/`

#### 3. Run Baseline Experiments (Optional)

```bash
python baselines/run_baseline_experiments_opt.py --grid_size 13 --experiment_type ALL_MODS --ratio 1_1
```

**Output:** Baseline results saved in `./baselines/data/grid{size}/`

#### 4. Process Data for Analysis

```bash
python prepare_data_for_analysis.py --grid_size 13 --wcd_pred_model_id {model_id}
```

**Output:** Processed data saved in `./summary_data/grid{size}/ml-our-approach/{model_id}/`

#### 5. Generate Plots and Analysis

```bash
python analyse_and_plot.py --grid_size 13 --wcd_pred_model_id {model_id} --time_out 600
```

**Parameters:**
- `--grid_size`: Grid size
- `--wcd_pred_model_id`: Model ID for analysis
- `--time_out`: Timeout threshold for data processing
- `--file_type`: Output file type (pdf, png)

**Output:** Plots saved in `./plots/grid{size}/{model_id}/`

## Experiment Types

### 1. BLOCKING_ONLY
- Only allows blocking actions (adding obstacles)
- Focuses on strategic placement of barriers

### 2. ALL_MODS
- Allows both blocking and unblocking actions in a ratio i.e 1_1 means 1:1 
- More flexible optimization with ratio constraints

### 3. BOTH_UNIFORM
- Uniform cost for both blocking and unblocking
- Balanced approach to environment modification

## Model Architectures

The framework supports multiple model types for WCD prediction:

- **CNN**: Convolutional Neural Network (default)
- **Transformer**: Vision Transformer
- **Linear**: Simple linear model
- **GNN**: Graph Neural Network
- **KRR**: Kernel Ridge Regression
- **GP**: Gaussian Process

## Key Features

### Optimization Algorithm
- Gradient-based optimization using trained WCD prediction models
- Constraint handling for valid environment modifications
- Multiple experiment types with different modification strategies

### Analysis Capabilities
- Comprehensive performance comparison between approaches
- Statistical analysis with standard errors
- Visualization of WCD reduction vs. modification budget
- Time complexity analysis

### Scalability
- Batch processing for multiple environment instances
- GPU acceleration for model training and optimization
- Configurable parameters for different grid sizes

## Output Files

### Training Outputs
- `models/wcd_prediction/grid{size}/training_logs/{model_id}/`: Trained models and training logs
- `model_performance.csv`: Training performance metrics
- `model_summary.csv`: Model summary statistics

### Optimization Outputs
- `wcd_optim_results/ml-our-approach/grid{size}/{model_id}/`: Optimization results
- JSON files with lambda pairs and performance metrics
- Environment modification data

### Analysis Outputs
- `plots/grid{size}/{model_id}/`: Generated plots
- `summary_data/grid{size}/`: Processed analysis data
- Performance comparison tables

## Troubleshooting

### Common Issues

1. **Missing Data**: Ensure training data exists in `data/grid{size}/model_training/`
2. **GPU Memory**: Reduce batch size or use CPU if GPU memory is insufficient
3. **Model Not Found**: Verify the `wcd_pred_model_id` matches a trained model
4. **Timeout Errors**: Increase `--time_out` parameter for longer experiments

### Debug Mode

Use debug mode for testing with smaller datasets:
```bash
python train_wcd_pred_model.py --run_mode debug
```

## Advanced Usage

### Custom Model Training
```bash
python train_wcd_pred_model.py --model_type transformer --epochs 200 --lr 0.001
```

### Sensitivity Analysis
```bash
python run_optimization_opt.py --sensitivity_analysis True --noise_level 0.001
```

### Batch Processing
Use the shell scripts in `scripts/` for batch processing multiple experiments.

## Dependencies

Ensure to install the packages in `../environment.yml`. We encourage using conda to setup the enviroment 