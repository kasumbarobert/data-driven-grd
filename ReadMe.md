# Goal Recognition Design for General Behavioral Agents using Machine Learning

This repository contains the code and data to reproduce results for the paper "Goal Recognition Design for General Behavioral Agents using Machine Learning". We provide code for minimizing Worst-Case Distance (WCD) in goal recognition design across different behavioral settings: optimal agents, suboptimal agents with hyperbolic discounting, and human behavior data.

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

## Experimental Settings

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

## Install dependencies

Install the required packages using conda:
```bash
conda env create -f environment.yml
conda activate gridworld-mdp
```

## Regenerate Paper Figures
To regenerate the figures in the paper, run the following corresponding commands. 


### Figure 3(a), 9(b) and 9(b)
```bash
cd optimal
bash regenerate_figure_3a_9a_9b.sh
```

### Figures 3(b), 4(a) and 4(b)
```bash
cd optimal
bash regenerate_figure_3b_4a_4b.sh
```

### Figure 5(b)
```bash
cd suboptimal
bash regenerate_figure_5b.sh
```

### Figures 5(a) and 8(d)

```bash
cd overcooked-ai/src/overcooked_ai_py/simulations
bash regenerate_paper_figure_5a_and_8d.sh
```

### Figure 6(a) and 6(b)
```bash
cd overcooked-ai/src/overcooked_ai_py/simulations
bash regenerate_figure_6a_and_6b.sh
```

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





