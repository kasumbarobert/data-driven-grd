# Human Behavior (Data-Driven) Experiments

This folder contains the code for the human-behavior setting in the paper.

## Overview

- `simulate_training_data_human_bhvr.py`: Simulate training data.
- `train_wcd_predictor_human_bhvr.py`: Train WCD predictor models.
- `run_optimization_human_bhvr.py`: Run optimization using trained predictors.
- `run_baseline_experiments_human_bhvr.py`: Run baseline methods.
- `prepare_results_for_analysis_human_bhvr.py`: Aggregate experiment outputs.
- `analyze_and_plot_human_bhvr.py`: Generate analysis plots.
- `validation-study/`: Scripts and assets for validation-study figures.

## Quick Start

From the repository root:

```bash
conda env create -f environment.yml
conda activate data-driven-grd
cd human-exp-data-driven
```

Typical workflow:

```bash
python simulate_training_data_human_bhvr.py
python train_wcd_predictor_human_bhvr.py
python run_optimization_human_bhvr.py
python run_baseline_experiments_human_bhvr.py
python prepare_results_for_analysis_human_bhvr.py
python analyze_and_plot_human_bhvr.py
```

## Related Data

Human subject data used by this setting is stored at:

- `../human-behavior-data/human-subject-experiment/`
