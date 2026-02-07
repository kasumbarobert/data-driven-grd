#!/bin/bash

# Regenerate paper figure 5(a) and 8(d)

python prepare_optim_results_for_visualization.py
python analysis/analyse_visualize_results.py 
