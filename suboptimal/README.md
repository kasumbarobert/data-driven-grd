# Parameterized Suboptimal Behavioral Assumption

This folder contains the implementation for an **Suboptimal Agents** with hyperbolic discounting behavioral agent in Gridworld environment.

## Overview

The framework consists of several key components:
- **MDP Implementation**: Core Gridworld MDP functionality with hyperbolic discounting (`mdp.py`)
- **Utility Functions**: Helper functions and model architectures for suboptimal agents (`utils_suboptimal.py`)
- **Data Generation**: Training data generation for suboptimal agents (`simulate_training_data_suboptimal.py`)
- **Model Training**: WCD prediction model training (`train_wcd_predictor_suboptimal.py`)
- **Optimization**: Main optimization experiments (`run_optimization_suboptimal.py`)
- **Baselines**: Comparison baseline experiments (`run_baseline_experiments_suboptimal.py`)

## Key Differences from Optimal Setting

- **Suboptimal Agents**: Uses hyperbolic discounting with parameter K instead of optimal agents

## Directory Structure

```
suboptimal/
├── mdp.py                           # Core MDP implementation with hyperbolic discounting
├── utils_suboptimal.py              # Utility functions for suboptimal agents
├── simulate_training_data_suboptimal.py      # Training data generation
├── train_wcd_predictor_suboptimal.py          # WCD prediction model training
├── run_optimization_suboptimal.py       # Main optimization experiments
├── run_baseline_experiments_suboptimal.py # Baseline experiments
├── run_exp_script_subopt.sh         # Shell script for batch execution
├── baselines/                       # Baseline experiment implementations
├── data/                           # Training and test data
├── models/                         # Trained models
├── plot_data/                      # Analysis data
└── plots/                          # Generated plots and visualizations
```

## Quick Start Guide

### Prerequisites

1. **Data**: Generate training data using `simulate_training_data_suboptimal.py` or ensure data exists in `data/grid{size}/model_training/`
2. **Dependencies**: Create the conda environment from the repository root `environment.yml`
3. **GPU**: Recommended for faster training and optimization

Environment setup (from repository root):
```bash
conda env create -f environment.yml
conda activate data-driven-grd
```

### Step-by-Step Execution

#### 1. Generate Training Data (if needed)

```bash
python simulate_training_data_suboptimal.py --grid_size 6 --K 4
```

**Parameters:**
- `--grid_size`: Size of the gridworld (default: 10)
- `--K`: Hyperbolic discounting parameter (default: 4)

**Output:** Training data saved in `./data/grid{size}/model_training/hyperbol_simulated_envs_K{K}_0.pkl`

#### 2. Train WCD Prediction Model

```bash
python train_wcd_predictor_suboptimal.py --grid_size 6 --K 4 --epochs 20 --batch_size 512 --lr 0.001
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
python run_optimization_suboptimal.py --grid_size 6 --experiment_type ALL_MODS --ratio 1_1 --wcd_pred_model_id {model_id} --num_instances 5 --max_iter 20 --K 4
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
python run_baseline_experiments_suboptimal.py --grid_size 6 --experiment_type ALL_MODS --ratio 1_1 --K 4
```

**Output:** Baseline results saved in `./baselines/data/grid{size}/`

#### 5. Prepare Results and Plot

```bash
python prepare_results_for_analysis_suboptimal.py
python analyze_and_plot_suboptimal.py
```

**Output:** Plots saved in `./plots/`

## Regenerate Paper Figures (Suboptimal Setting)

Recommended (from repository root):

```bash
python manuscript_figures/generate_all.py 5b 8c
```

Outputs are written to `manuscript_figures/generated_figures/`.

Local script (from `suboptimal/`) for Figure 5(b):

```bash
bash regenerate_figure_5b.sh
```

## Experiment Types

### 1. BLOCKING_ONLY
- Only allows blocking actions (adding obstacles)

### 2. ALL_MODS
- Allows both blocking and unblocking actions in a ratio (e.g., 1_1 means 1:1)

### 3. BOTH_UNIFORM
- Uniform cost for both blocking and unblocking

## Hyperbolic Discounting Parameter (K)

The K parameter controls the degree of suboptimality in agent behavior:
- **Lower K values**: More suboptimal agents (shorter planning horizons)
- **Higher K values**: More optimal agents (longer planning horizons)
- **Typical range**: 0.1 to 10.0

## Model Architecture

The suboptimal framework uses a simplified model architecture:
- **CNN4**: Convolutional Neural Network with ResNet18 backbone
- Optimized for suboptimal agent WCD prediction



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
