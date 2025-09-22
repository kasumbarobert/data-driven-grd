# Data-Driven Goal Recognition Design Application to Overcooked-AI domain

Extension of our data-driven goal recognition design approach to the Overcooked-AI domain.

## Prerequisites

- Python 3.10.6
- GPU with CUDA

## Workflow

### Step 1: Generate Training Data

Generate Overcooked-AI environment designs and compute WCD values.

```bash
python /src/overcooked_ai_py/simulations/simulate_wcd_oracle.py --simulation-type random --max_grid_size 6
```

**Output**: Training data in `data/grid{size}/model_training/`

### Step 2: Train CNN Oracle

Train a CNN to predict WCD values for environment designs.

```bash
python /src/overcooked_ai_py/simulations/train_model.py --train --grid_size 6
```

**Output**: Trained model saved as `models/wcd_nn_oracle_{grid_size}.pt`


### Step 3: Run Optimization

Use the trained CNN to optimize environments and minimize WCD.

```bash
python /src/overcooked_ai_py/simulations/run_optmize_wcd.py --cost 0 --start_index 0 --max_grid_size 6 --experiment_label test --optimality OPTIMAL --experiment_type CONSTRAINED
```

**Output**: Results in `data/grid{size}/optim_runs/{experiment_type}/langrange_values/`

### Step 4: Run Baseline Experiments

Generate baseline results for comparison.

```bash
python /src/overcooked_ai_py/simulations/run_baseline_experiments.py --cost 10 --max_grid_size 6 --experiment_label test --experiment_type GREEDY_TRUE_WCD --optimality OPTIMAL --start_index 0 --timeout_seconds 18000 --ratio 1_3
```

**Output**: Baseline results in `./src/overcooked_ai_py/baselines`

### Step 5: Prepare Results for Visualization

Process optimization results for analysis.

```bash
python /src/overcooked_ai_py/simulations/prepare_optim_results_for_visualization.py
```

**Output**: Processed data in CSV format

### Step 6: Analyze and Visualize Results

Generate analysis and visualizations.

```bash
python /src/overcooked_ai_py/simulations/analysis/analyse_visualize_results.py
```

**Output**: Plots and analysis in `./plots/` directory

## REGENERATE PAPER FIGURE 5(a) and 8(d)

To regenerate the two figures from the paper, ensure that you have the provided data folders with experimental results, then run:

```bash
cd /src/overcooked_ai_py/simulations
bash regenerate_paper_figure_5a_and_8d.sh
```

**What this does:**
1. **prepare_optim_results_for_visualization.py**: Processes raw optimization results and aggregates them across environments and budget levels
2. **analyse_visualize_results.py**: Generates publication-quality plots comparing our approach against baseline methods

**Output:**
- **Figure 5(a)**: Time comparison plot saved as `./plots/time/overcooked_time.pdf`
- **Figure 8(d)**: WCD reduction plot saved as `./plots/wcd_reduction/overcooked_wcd_reduction.pdf`

**Prerequisites:**
- Experimental results from Steps 3-4 must be available in the expected data directories
- Baseline results in `./baselines/data/grid6/optim_runs/timeout_18000/`
- Optimization results in `../data/grid6/optim_runs/CONSTRAINED/ratio_0_0/n_lambdas_17/`


## Reference

This code is adapted from the original [Overcooked-AI](https://github.com/HumanCompatibleAI/human_aware_rl) repository. The original project provides the foundational tools and algorithms for reinforcement learning and human-aware AI in the Overcooked game environment.

