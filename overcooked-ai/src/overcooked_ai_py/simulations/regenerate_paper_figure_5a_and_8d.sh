#!/bin/bash

# Regenerate paper figure 5(a) and 8(d)

python prepare_results_for_analysis_overcooked.py
python analysis/analyze_and_plot_overcooked.py 
