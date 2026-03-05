# Optimal Behavioral Assumption

This folder contains the implementation for an optimal behavioral agent in Gridworld environment.

## Overview

The framework consists of several key components:
- **MDP Implementation**: Core Gridworld MDP functionality (`mdp.py`)
- **Utility Functions**: Helper functions and model architectures (`utils.py`)
- **Model Training**: WCD prediction model training (`train_wcd_predictor_optimal.py`)
- **Optimization**: Main optimization experiments (`run_optimization_optimal.py`)
- **Baselines**: Comparison baseline experiments (`baselines/`)
- **Analysis**: Data processing and visualization (`prepare_results_for_analysis_optimal.py`, `analyze_and_plot_optimal.py`)

## Directory Structure

```
optimal/
├── mdp.py                           # Core MDP implementation
├── utils.py                         # Utility functions and model architectures
├── train_wcd_predictor_optimal.py          # WCD prediction model training
├── run_optimization_optimal.py          # Main optimization experiments
├── prepare_results_for_analysis_optimal.py     # Data processing for analysis
├── analyze_and_plot_optimal.py              # Visualization and final analysis
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
2. **Dependencies**: Create the conda environment from the repository root `environment.yml`
3. **GPU**: Recommended for faster training and optimization

Environment setup (from repository root):
```bash
conda env create -f environment.yml
conda activate data-driven-grd
```

### Step-by-Step Execution

#### 1. Train WCD Prediction Model

```bash
python train_wcd_predictor_optimal.py --grid_size 13 --model_type cnn --epochs 100 --batch_size 512 --lr 0.01
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
python run_optimization_optimal.py --grid_size 13 --experiment_type ALL_MODS --ratio 1_1 --wcd_pred_model_id {model_id} --num_instances 5 --max_iter 20
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
python baselines/run_baseline_experiments_optimal.py --grid_size 13 --experiment_type ALL_MODS --ratio 1_1
```

**Output:** Baseline results saved in `./baselines/data/grid{size}/`

#### 4. Process Data for Analysis

```bash
python prepare_results_for_analysis_optimal.py --grid_size 13 --wcd_pred_model_id {model_id}
```

**Output:** Processed data saved in `./summary_data/grid{size}/ml-our-approach/{model_id}/`

#### 5. Generate Plots and Analysis

```bash
python analyze_and_plot_optimal.py --grid_size 13 --wcd_pred_model_id {model_id} --time_out 600
```

**Parameters:**
- `--grid_size`: Grid size
- `--wcd_pred_model_id`: Model ID for analysis
- `--time_out`: Timeout threshold for data processing
- `--file_type`: Output file type (pdf, png)

**Output:** Plots saved in `./plots/grid{size}/{model_id}/`

## Regenerate Paper Figures (Optimal Setting)

Recommended (from repository root):

```bash
python manuscript_figures/generate_all.py 3a 3b 4a 4b 8a 8b 9a 9b
```

Outputs are written to `manuscript_figures/generated_figures/`.

Local scripts (from `optimal/`) for the main-text figures:

```bash
bash regenerate_figure_3a_9a_9b.sh
bash regenerate_figure_3b_4a_4b.sh
```

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
