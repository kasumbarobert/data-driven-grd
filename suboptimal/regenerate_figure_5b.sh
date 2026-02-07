#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" prepared_data_for_analysis.py
"$PYTHON_BIN" result_analysis_suboptimal.py
