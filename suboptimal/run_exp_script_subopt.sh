#!/bin/bash

# Set default experiment labels
default_labels=("BLOCKING_ONLY_GREEDY_TRUE_WCD" "BLOCKING_ONLY_GREEDY_PRED_WCD" "BOTH_UNIFORM_GREEDY_TRUE_WCD" "BOTH_UNIFORM_GREEDY_PRED_WCD")

# Set default timeout values for different grid sizes
default_grid_10_timeouts=("600")
default_grid_6_timeouts=("300" )

# Initialize variables with default values
experiment_labels=("${default_labels[@]}")

timeouts=("${default_grid_10_timeouts[@]}")
grid_size=10

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --labels)
            shift
            experiment_labels=($1)
            ;;
        --timeouts)
            shift
            timeouts=($@)
            break
            ;;
        --grid_size)
            shift
            grid_size=$1
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

# Check if the grid size is 6 and update the default timeouts
if [ "$grid_size" -eq 6 ]; then
    timeouts=("${default_grid_6_timeouts[@]}")
fi

# Iterate over timeouts and run the Python script with nested experiment labels
for timeout in "${timeouts[@]}"
do
    echo "Running experiments with timeout $timeout seconds, interval $interval, and grid size $grid_size:"
    # Iterate over the experiment labels
    for label in "${experiment_labels[@]}"
    do
        echo "Running experiment: $label"
        python3 baseline_experiments_subopt.py --experiment_type $label --grid_size "$grid_size"  --timeout_seconds "$timeout"
        echo "Experiment $label (Timeout $timeout) completed."
    done
done

echo "All experiments completed."
