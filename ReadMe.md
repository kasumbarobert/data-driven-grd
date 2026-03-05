# Goal Recognition Design for General Behavioral Agents using Machine Learning

This repository contains the code and data to reproduce results for the paper []"Goal Recognition Design for General Behavioral Agents using Machine Learning"](https://openreview.net/forum?id=GDuWBhvMid). We provide code for modifying environment designs to mininimize the Worst-Case Distinctiveness (WCD) using our Lagrange-based Gradient Descent approach. The provided code covers the different behavioral settings in our paper: optimal agents, suboptimal agents with hyperbolic discounting, and human behavior data.

## Overview

The repository provides:
- **Simulated data generation** for environments with corresponding WCD values
- **WCD prediction model training** scripts
- **Optimization code** using trained WCD predictive models to modify environments
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
│   ├── train_wcd_predictor_optimal.py # WCD prediction model training
│   ├── run_optimization_optimal.py # Main optimization experiments
│   ├── baselines/             # Baseline experiment implementations
│   └── models/                # Trained models for grid sizes 6 and 13
│
├── suboptimal/                # Suboptimal agent experiments
│   ├── README.md              # Detailed guide for suboptimal setting
│   ├── train_wcd_predictor_suboptimal.py # WCD prediction model training
│   ├── run_optimization_suboptimal.py # Main optimization experiments
│   ├── simulate_training_data_suboptimal.py # Training data generation
│   └── models/                # Trained models for grid size 6
│
├── human-exp-data-driven/     # Human behavior experiments
│   ├── README.md              # Detailed guide for human behavior setting
│   ├── run_optimization_human_bhvr.py # Main optimization experiments
│   └── baselines/             # Baseline experiment implementations
│
├── overcooked-ai/             # Overcooked-AI domain extension
│   ├── README.md              # Detailed guide for Overcooked-AI
│   └── src/                   # Implementation code
│
└── human-behavior-data/       # MTurk human subject experiment data
```

## Experimental Settings

**Optimal Agents**: For environments with optimal agent behavior
- **Use case**: Standard gridworld optimization with optimal pathfinding
- **Key features**: Multiple model architectures (CNN, Transformer, GNN, etc.)
- **See**: `optimal/README.md` for detailed instructions

**Suboptimal Agents**: For environments with hyperbolic discounting agents
- **Use case**: Realistic agent behavior with time discounting
- **Key features**: K parameter controls suboptimality level
- **See**: `suboptimal/README.md` for detailed instructions

**Human Behavior**: For data-driven human behavior experiments
- **Use case**: Real human behavior from MTurk experiments
- **Key features**: Uses actual human trajectory data
- **See**: `human-exp-data-driven/README.md` for detailed instructions

**Overcooked-AI**: For the Overcooked-AI domain
- **Use case**: Multi-agent coordination scenarios
- **Key features**: Domain-specific implementations
- **See**: `overcooked-ai/README.md` for detailed instructions

## Install dependencies

Install the required packages using the repository’s `environment.yml`:
```bash
conda env create -f environment.yml
conda activate data-driven-grd
```

## Regenerate Paper Figures
Use the unified figure driver from the repository root:

```bash
python manuscript_figures/generate_all.py --list
```

Generate specific figures:

```bash
python manuscript_figures/generate_all.py 3a 3b 4a 4b 5a 5b 6a 6b 8a 8b 8c 8d 9a 9b
```

Generate all available figure drivers:

```bash
python manuscript_figures/generate_all.py
```

Generated outputs are saved in `manuscript_figures/generated_figures/`.

Notes:
- Use an environment where `python` has the project dependencies (especially PyTorch) if you want full recomputation from raw data.

## Citation

If you use this framework in your research, please cite the following.

``` 
@article{kasumba2025data,
  title={Data-driven goal recognition design for general behavioral agents},
  author={Kasumba, Robert and Yu, Guangyu and Ho, Chien-Ju and Keren, Sarah and Yeoh, William},
  journal={Transactions in Machine Learning Research},
  year={2025}
}
```
