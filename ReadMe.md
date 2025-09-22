# Data-Driven Goal Recognition Design for General Behavioral Agents

This repository contains the code "Goal Recognition Design for General Behavioral Agents using Machine Learning". We provide code for minimizing Worst-Case Distance (WCD) in goal recognition design across different behavioral settings: optimal agents, suboptimal agents with hyperbolic discounting, and human behavior data.

## Overview

The repository provides:
- **Simulated data generation** for environments with corresponding WCD values
- **WCD prediction model training** with detailed hyperparameters and notebooks
- **Optimization code** using trained WCD predictive models to minimize environment WCD
- **Baseline implementations** including Greedy, Exhaustive Search, and Pruned-Reduce methods
- **Multiple behavioral settings**: optimal agents, suboptimal agents, and human behavior data
- **Overcooked-AI domain extension** with specific implementations

## Prerequisites

- **Python**: Python 3.10.6 or higher
- **CUDA**: GPU support recommended for faster training and optimization
- **Dependencies**: Install packages from `environment.yml` using conda

## Repository Structure

```
├── optimal/                    # Optimal agent experiments
│   ├── README.md              # Detailed guide for optimal setting
│   ├── train_wcd_pred_model.py # WCD prediction model training
│   ├── run_optimization_opt.py # Main optimization experiments
│   ├── baselines/             # Baseline experiment implementations
│   └── models/                # Trained models for grid sizes 6 and 13
│
├── suboptimal/                # Suboptimal agent experiments
│   ├── README.md              # Detailed guide for suboptimal setting
│   ├── train_wcd_pred_model.py # WCD prediction model training
│   ├── run_optimization_subopt.py # Main optimization experiments
│   ├── simulate_data_generation.py # Training data generation
│   └── models/                # Trained models for grid size 6
│
├── human-exp-data-driven/     # Human behavior experiments
│   ├── README.md              # Detailed guide for human behavior setting
│   ├── run_optimization_data_driven.py # Main optimization experiments
│   └── baselines/             # Baseline experiment implementations
│
├── overcooked-ai/             # Overcooked-AI domain extension
│   ├── README.md              # Detailed guide for Overcooked-AI
│   └── src/                   # Implementation code
│
└── human-behavior-data/       # MTurk human subject experiment data
```

## Quick Start Guide

### 1. Choose Your Experimental Setting

**Optimal Agents**: For environments with optimal agent behavior
- **Use case**: Standard gridworld optimization with optimal pathfinding
- **Key features**: Multiple model architectures (CNN, Transformer, GNN, etc.)
- **Grid sizes**: 6 and 13 supported
- **See**: `optimal/README.md` for detailed instructions

**Suboptimal Agents**: For environments with hyperbolic discounting agents
- **Use case**: Realistic agent behavior with time discounting
- **Key features**: K parameter controls suboptimality level
- **Grid sizes**: 6 and 10 supported
- **See**: `suboptimal/README.md` for detailed instructions

**Human Behavior**: For data-driven human behavior experiments
- **Use case**: Real human behavior from MTurk experiments
- **Key features**: Uses actual human trajectory data
- **Grid sizes**: 6 and 13 supported
- **See**: `human-exp-data-driven/README.md` for detailed instructions

**Overcooked-AI**: For the Overcooked-AI domain
- **Use case**: Multi-agent coordination scenarios
- **Key features**: Domain-specific implementations
- **See**: `overcooked-ai/README.md` for detailed instructions

### 2. Complete Sequential Workflow

The framework follows a comprehensive sequential process from data generation to final analysis and visualization. Here's the complete workflow for each experimental setting:

#### Optimal Setting - Complete Workflow

**Step 1: Train WCD Prediction Model**
```bash
# Train WCD prediction model
python optimal/train_wcd_pred_model.py --grid_size 6 --model_type cnn --epochs 100
```
- **Purpose**: Train a machine learning model to predict WCD values
- **Output**: Trained model saved in `models/wcd_prediction/grid{size}/training_logs/{model_id}/`
- **Key Features**: Multiple architectures (CNN, Transformer, GNN, etc.)

**Step 2: Run Optimization Experiments**
```bash
# Run optimization using trained model
python optimal/run_optimization_opt.py --grid_size 6 --experiment_type ALL_MODS --ratio 1_1 --wcd_pred_model_id {model_id}
```
- **Purpose**: Use trained model to optimize environments and minimize WCD
- **Output**: Optimization results in `wcd_optim_results/ml-our-approach/grid{size}/{model_id}/`
- **Key Features**: Gradient-based optimization with constraint handling

**Step 3: Run Baseline Experiments**
```bash
# Run baseline comparisons
python optimal/baselines/run_baseline_experiments_opt.py --experiment BLOCKING_ONLY_GREEDY_PRED_WCD --grid_size 6
```
- **Purpose**: Generate baseline results for comparison
- **Output**: Baseline results in `baselines/data/grid{size}/`

**Step 4: Prepare Data for Analysis**
```bash
# Process optimization results for analysis
python optimal/prepare_data_for_analysis.py --grid_size 6 --wcd_pred_model_id {model_id}
```
- **Purpose**: Process raw optimization results into analysis-ready format
- **Output**: Processed data in `summary_data/grid{size}/ml-our-approach/{model_id}/`
- **Key Features**: Aggregates results across environments and budget levels

**Step 5: Analyze and Visualize Results**
```bash
# Generate comprehensive analysis and plots
python optimal/analyse_and_plot.py --grid_size 6 --wcd_pred_model_id {model_id}
```
- **Purpose**: Create visualizations and statistical analysis
- **Output**: Plots and analysis in `plots/` directory
- **Key Features**: Performance comparisons, WCD reduction analysis, time complexity plots

#### Suboptimal Setting - Complete Workflow

**Step 1: Generate Training Data**
```bash
# Generate training data for suboptimal agents
python suboptimal/simulate_data_generation.py --grid_size 6 --K 4
```
- **Purpose**: Create training data with hyperbolic discounting behavior
- **Output**: Training data in `data/grid{size}/model_training/hyperbol_simulated_envs_K{K}_0.pkl`
- **Key Features**: Simulates suboptimal agent behavior with parameter K

**Step 2: Train WCD Prediction Model**
```bash
# Train model for suboptimal agents
python suboptimal/train_wcd_pred_model.py --grid_size 6 --K 4 --epochs 20
```
- **Purpose**: Train model to predict WCD for suboptimal agents
- **Output**: Trained model in `models/wcd_prediction/grid{size}/training_logs/{model_id}/`
- **Key Features**: CNN4 with ResNet18 backbone, hyperbolic Q-function

**Step 3: Run Optimization Experiments**
```bash
# Run optimization for suboptimal setting
python suboptimal/run_optimization_subopt.py --grid_size 6 --experiment_type ALL_MODS --K 4 --wcd_pred_model_id {model_id}
```
- **Purpose**: Optimize environments for suboptimal agent behavior
- **Output**: Optimization results in `wcd_optim_results/ml-our-approach/grid{size}/{model_id}/`

**Step 4: Run Baseline Experiments**
```bash
# Run baseline comparisons
python suboptimal/run_baseline_experiments_subopt.py --experiment BLOCKING_ONLY_GREEDY_PRED_WCD --grid_size 6 --K 4
```
- **Purpose**: Generate baseline results for suboptimal setting
- **Output**: Baseline results in `baselines/data/grid{size}/`

**Step 5: Prepare and Analyze Results**
```bash
# Process and visualize suboptimal results
python suboptimal/prepare_data_for_analysis.py --grid_size 6 --K 4 --wcd_pred_model_id {model_id}
python suboptimal/analyse_and_plot.py --grid_size 6 --K 4 --wcd_pred_model_id {model_id}
```
- **Purpose**: Process results and create visualizations for suboptimal setting
- **Output**: Analysis plots and processed data

#### Human Behavior Setting - Complete Workflow

**Step 1: Run Optimization with Human Data**
```bash
# Run optimization using human behavior data
python human-exp-data-driven/run_optimization_data_driven.py --grid_size 6
```
- **Purpose**: Optimize environments using real human behavior data
- **Output**: Optimization results using human trajectories

**Step 2: Run Baseline Experiments**
```bash
# Run baseline comparisons
python human-exp-data-driven/run_baseline_experiments_data_driven.py --experiment BLOCKING_ONLY_GREEDY_PRED_WCD --grid_size 6
```
- **Purpose**: Generate baseline results for human behavior setting
- **Output**: Baseline results for comparison

**Step 3: Prepare and Analyze Results**
```bash
# Process and visualize human behavior results
python human-exp-data-driven/prepare_data_for_analysis.py --grid_size 6
python human-exp-data-driven/analyse_and_plot.py --grid_size 6
```
- **Purpose**: Analyze results from human behavior experiments
- **Output**: Analysis plots and performance metrics

#### Overcooked-AI Setting - Complete Workflow

**Step 1: Run Optimization**
```bash
# Run optimization in Overcooked-AI domain
python overcooked-ai/src/overcooked_ai_py/simulations/run_optimize_wcd.py --cost COST --start_index START_INDEX --max_grid_size MAX_GRID_SIZE --experiment_label EXPERIMENT_LABEL --optimality OPTIMALITY --experiment_type EXPERIMENT_TYPE
```
- **Purpose**: Optimize environments in multi-agent Overcooked-AI domain
- **Output**: Optimization results in `./results` directory

**Step 2: Run Baseline Experiments**
```bash
# Run baseline comparisons
python overcooked-ai/src/overcooked_ai_py/simulations/run_baseline_experiments.py --cost COST --max_grid_size MAX_GRID_SIZE --experiment_label EXPERIMENT_LABEL --experiment_type EXPERIMENT_TYPE --optimality OPTIMALITY --start_index START_INDEX --timeout_seconds TIMEOUT_SECONDS --ratio RATIO
```
- **Purpose**: Generate baseline results for Overcooked-AI domain
- **Output**: Baseline results for comparison

**Step 3: Prepare Results for Visualization**
```bash
# Process optimization results for visualization
python overcooked-ai/src/overcooked_ai_py/simulations/prepare_optim_results_for_visualization.py
```
- **Purpose**: Process raw optimization results into visualization-ready format
- **Output**: Processed data in CSV format for analysis

**Step 4: Analyze and Visualize Results**
```bash
# Generate comprehensive analysis and plots
python overcooked-ai/src/overcooked_ai_py/simulations/analysis/analyse_visualize_results.py
```
- **Purpose**: Create visualizations and statistical analysis for Overcooked-AI results
- **Output**: Plots and analysis in `./plots/` directory
- **Key Features**: Performance comparisons, WCD reduction analysis, time complexity plots

## Data Preparation and Analysis Tools

### Data Preparation Process

The data preparation step is crucial for converting raw optimization results into analysis-ready formats:

**What it does:**
- **Aggregates Results**: Combines optimization results across multiple environments and budget levels
- **Processes Lambda Values**: Organizes results by different lambda parameter combinations
- **Handles Timeouts**: Manages experiments that didn't complete within time limits
- **Creates Summary Statistics**: Generates mean, standard error, and other statistical measures
- **Formats for Analysis**: Converts data into CSV format suitable for visualization tools

**Key Scripts:**
- `prepare_data_for_analysis.py` - Main data processing script
- `prepare_optim_results_for_visualization.py` - Overcooked-AI specific processing

### Analysis and Visualization Process

The analysis step provides comprehensive insights into optimization performance:

**What it does:**
- **Performance Comparison**: Compares our approach against baseline methods
- **Statistical Analysis**: Computes significance tests and confidence intervals
- **WCD Reduction Analysis**: Measures effectiveness of environment modifications
- **Time Complexity Analysis**: Evaluates computational efficiency
- **Budget Analysis**: Analyzes relationship between modification budget and performance

**Generated Visualizations:**
- **Performance Plots**: WCD reduction vs. budget curves with error bars
- **Time Analysis**: Optimization time vs. budget plots
- **Comparison Charts**: Side-by-side comparison of different approaches
- **Statistical Plots**: Confidence intervals and significance testing results

**Key Scripts:**
- `analyse_and_plot.py` - Main analysis and visualization script
- `analyse_visualize_results.py` - Overcooked-AI specific analysis

## Key Features by Setting

### Optimal Setting
- **Model Architectures**: CNN, Transformer, Linear, GNN, KRR, GP
- **Experiment Types**: BLOCKING_ONLY, ALL_MODS, BOTH_UNIFORM
- **Optimization**: Gradient-based with constraint handling
- **Analysis**: Comprehensive performance comparison and visualization

### Suboptimal Setting
- **Hyperbolic Discounting**: K parameter controls agent suboptimality
- **Model Architecture**: CNN4 with ResNet18 backbone
- **Data Generation**: Built-in data generation for suboptimal scenarios
- **K Values**: 0.1 to 10.0 (lower = more suboptimal)

### Human Behavior Setting
- **Real Data**: Uses MTurk human subject experiment data
- **Data-Driven**: Trained on actual human trajectories
- **Validation**: Real human behavior validation

### Overcooked-AI Setting
- **Multi-Agent**: Coordination scenarios
- **Domain-Specific**: Custom implementations for Overcooked-AI
- **Cost Functions**: Various cost structures

## Experiment Types

### BLOCKING_ONLY
- Only allows blocking actions (adding obstacles)
- Focuses on strategic barrier placement

### ALL_MODS
- Allows both blocking and unblocking actions
- Ratio constraints (e.g., 1:1, 3:1, 1:3)
- More flexible optimization

### BOTH_UNIFORM
- Uniform cost for both blocking and unblocking
- Balanced approach to environment modification

## Output and Results

### Training Outputs
- **Models**: Saved in `models/wcd_prediction/grid{size}/training_logs/{model_id}/`
- **Logs**: Training curves, performance metrics, detailed logs
- **Visualizations**: Loss curves, performance plots

### Optimization Outputs
- **Results**: Saved in `wcd_optim_results/ml-our-approach/grid{size}/{model_id}/`
- **Metrics**: WCD reduction, modification budgets, time complexity
- **JSON Files**: Lambda pairs and performance metrics

### Baseline Outputs
- **Results**: Saved in `baselines/data/grid{size}/`
- **Comparisons**: Performance vs. our approach

## Troubleshooting

### Common Issues
1. **Missing Data**: Generate data using appropriate scripts
2. **GPU Memory**: Reduce batch size or use CPU
3. **Model Not Found**: Verify model IDs match trained models
4. **K Parameter**: Ensure consistency across training and optimization (suboptimal)

### Debug Mode
```bash
# Optimal
python optimal/train_wcd_pred_model.py --run_mode debug

# Suboptimal
python suboptimal/train_wcd_pred_model.py --run_mode debug
```

## Advanced Usage

### Custom Parameters
```bash
# Optimal with custom model
python optimal/train_wcd_pred_model.py --model_type transformer --epochs 200 --lr 0.001

# Suboptimal with different K
python suboptimal/train_wcd_pred_model.py --K 8 --grid_size 6 --epochs 50

# Human behavior with custom settings
python human-exp-data-driven/run_optimization_data_driven.py --grid_size 13
```

### Batch Processing
Use shell scripts in each folder for batch processing:
```bash
bash optimal/scripts/run_optim_shell_script_bjob_array.sh
bash suboptimal/run_exp_script_subopt.sh
```

## REGENERATE PAPER FIGURES

### 3(a)

### 3(b)

### 4(a)

### 4(b)

### 5(b)


### 5(a) and 8(d)

To regenerate the two figures from the paper, ensure that you have the provided data folders with experimental results, then run:

```bash
cd overcooked-ai/src/overcooked_ai_py/simulations
bash regenerate_paper_figure_5a_and_8d.sh
```

### 6(a) and 6(b)



## Dependencies

Install the required packages using conda:
```bash
conda env create -f environment.yml
conda activate gridworld-mdp
```

## Citation

If you use this framework in your research, please cite the following.
@article{kasumba2025data,
  title={Data-driven goal recognition design for general behavioral agents},
  author={Kasumba, Robert and Yu, Guangyu and Ho, Chien-Ju and Keren, Shlomo and Yeoh, William},
  journal={Transactions in Machine Learning Research},
  year={2025}
}

## Getting Help

- **Optimal Setting**: See `optimal/README.md` for detailed instructions
- **Suboptimal Setting**: See `suboptimal/README.md` for detailed instructions  
- **Human Behavior**: See `human-exp-data-driven/README.md` for detailed instructions
- **Overcooked-AI**: See `overcooked-ai/README.md` for detailed instructions

Each folder contains comprehensive documentation specific to that experimental setting.





