# Data-Driven Goal Recognition Design Application to Overcooked-AI domain

Extension of our data-driven goal recognition design approach to the Overcooked-AI domain.

## Prerequisites

- Python 3.10.6
- GPU with CUDA
- Conda environment from repository root `environment.yml`

Environment setup (from repository root):
```bash
conda env create -f environment.yml
conda activate data-driven-grd
```

## Workflow

### Step 1: Generate Training Data

Generate Overcooked-AI environment designs and compute WCD values.

```bash
python src/overcooked_ai_py/simulations/simulate_training_data_overcooked.py --simulation-type random --max_grid_size 6
```

**Output**: Training data in `data/grid{size}/model_training/`

### Step 2: Train CNN Oracle

Train a CNN to predict WCD values for environment designs.

```bash
python src/overcooked_ai_py/simulations/train_wcd_predictor_overcooked.py --train --grid_size 6
```

**Output**: Trained model saved as `models/wcd_nn_oracle_{grid_size}.pt`


### Step 3: Run Optimization

Use the trained CNN to optimize environments and minimize WCD.

```bash
python src/overcooked_ai_py/simulations/run_optimization_overcooked.py --cost 0 --start_index 0 --max_grid_size 6 --experiment_label test --optimality OPTIMAL --experiment_type CONSTRAINED
```

**Output**: Results in `data/grid{size}/optim_runs/{experiment_type}/langrange_values/`

### Step 4: Run Baseline Experiments

Generate baseline results for comparison.

```bash
python src/overcooked_ai_py/simulations/run_baseline_experiments_overcooked.py --cost 10 --max_grid_size 6 --experiment_label test --experiment_type GREEDY_TRUE_WCD --optimality OPTIMAL --start_index 0 --timeout_seconds 18000 --ratio 1_3
```

**Output**: Baseline results in `./src/overcooked_ai_py/baselines`

### Step 5: Prepare Results for Visualization

Process optimization results for analysis.

```bash
python src/overcooked_ai_py/simulations/prepare_results_for_analysis_overcooked.py
```

**Output**: Processed data in CSV format

### Step 6: Analyze and Visualize Results

Generate analysis and visualizations.

```bash
python src/overcooked_ai_py/simulations/analysis/analyze_and_plot_overcooked.py
```

**Output**: Plots are written to `./plots/` relative to your current working directory.

## Regenerate Paper Figures 5(a) and 8(d)

Recommended (from repository root):

```bash
python manuscript_figures/generate_all.py 5a 8d
```

Local Overcooked-only flow (from `overcooked-ai/`):

```bash
cd src/overcooked_ai_py/simulations
bash regenerate_paper_figure_5a_and_8d.sh
```

**What this does:**
1. **prepare_results_for_analysis_overcooked.py**: Processes raw optimization results and aggregates them across environments and budget levels
2. **analyze_and_plot_overcooked.py**: Generates publication-quality plots comparing our approach against baseline methods

**Output:**
- Using `manuscript_figures/generate_all.py`: final artifacts in `manuscript_figures/generated_figures/Figure_5a.pdf` and `manuscript_figures/generated_figures/Figure_8d.pdf`
- Using the local shell script from `src/overcooked_ai_py/simulations/`: plots in `./plots/wcd_reduction/overcooked_wcd_reduction.pdf` and `./plots/time/overcooked_time.pdf`

**Prerequisites:**
- Experimental results from Steps 3-4 must be available in the expected data directories
- Baseline results in `./baselines/data/grid6/optim_runs/timeout_18000/`
- Optimization results in `../data/grid6/optim_runs/CONSTRAINED/ratio_0_0/n_lambdas_17/`


## Reference

This code is adapted from the original [Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai/) repository. The original project provides the foundational tools and algorithms for reinforcement learning and human-aware AI in the Overcooked game environment.
