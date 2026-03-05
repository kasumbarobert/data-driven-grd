#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" prepare_results_for_analysis_suboptimal.py
"$PYTHON_BIN" analyze_and_plot_suboptimal.py
