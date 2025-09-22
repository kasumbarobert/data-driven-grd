#!/bin/bash

# Set default experiment types
default_types=("GREEDY_TRUE_WCD","GREEDY_PRED_WCD")


default_grid_6_timeouts=("900" "1800" "3600" "14400" "7200")

# Initialize variables with default values
experiment_types=("${default_types[@]}")
interval=40
timeouts=("${default_grid_6_timeouts[@]}")
grid_size=6

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --types)
            shift
            experiment_types=($1)
            ;;
        --interval)
            shift
            interval=$1
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

# Iterate over timeouts and run the Python script with nested experiment types
for timeout in "${timeouts[@]}"
do
    echo "Running experiments with timeout $timeout seconds, interval $interval, and grid size $grid_size:"
    
    # Iterate over the experiment types
    for type in "${experiment_types[@]}"
    do
        echo "Running experiment: $type"
        python3 baseline_experiments.py --experiment_type $type --max_grid_size "$grid_size"  --timeout_seconds "$timeout" --optimality OPTIMAL
        echo "Experiment $type (Timeout $timeout) completed."
    done
done

echo "All experiments completed."
