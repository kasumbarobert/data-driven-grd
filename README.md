# data-driven-grd

This repository has the code implementation of our approach for gradient descent based GRD and the baselines. We also provide the final trained models and the dataset we used during the design modifications to minimize wcd for goal-recognition-design with optimal human behavior in gridworld domain.  

## Preriquistes
This project is based on python3 and assumes you have Python 3.10.6.

## Initializations

By default this codebase uses CPU, but if you have a GPU you can change that in `config.json` file 


## Our approach

To run our method, use the following command:

```bash
python gridworld/optimal/optimize_wcd.py --experiment EXPERIMENT_TYPE --grid_size GRID_SIZE --num_instances NUM_INSTANCES --timeout_seconds TIMEOUT_SECONDS
```

Replace the placeholders (`EXPERIMENT_TYPE`, `GRID_SIZE`, `NUM_INSTANCES`, and `TIMEOUT_SECONDS`) with your desired values.

### Arguments

- `experiment`: Specify the type of experiment to run - "BOTH_UNIFORM" allows both blocking and unblocking modifications. Choose from the following options: 
  - "BLOCKING_ONLY"
  - "BOTH_UNIFORM"

- `grid_size`: Set the maximum grid size. Choose from either 6 or 10.

- `num_instances`: Specify the number of Number of problem instances to run.

- `timeout_seconds`: Set the timeout in seconds for the experiments.

### Example

Here's an example command to run a blocking-only experiment with a grid size of 10, 500 instances, and a timeout of 600 seconds:

```bash
python gridworld/optimal/optimize_wcd.py --experiment BLOCKING_ONLY --grid_size 10 --num_instances 500 --timeout_seconds 600
```

Feel free to explore different experiment types and configurations by adjusting the arguments accordingly.

### Results

After running the experiments, the results will be stored in  and `./baselines/data` directories. Check these directories for experiment-specific data files.



## Baselines

To run baseline experiments, use the following command:

```bash
python gridworld/optimal/baseline_experiments.py --experiment EXPERIMENT_TYPE --grid_size GRID_SIZE --num_instances NUM_INSTANCES --timeout_seconds TIMEOUT_SECONDS
```

Replace the placeholders (`EXPERIMENT_TYPE`, `GRID_SIZE`, `NUM_INSTANCES`, and `TIMEOUT_SECONDS`) with your desired values.

### Arguments

- `experiment`: Specify the type of experiment to run - "BOTH_UNIFORM" allows both blocking and unblocking modifications. Choose from the following options: 
  - "BLOCKING_ONLY_EXHAUSTIVE"
  - "BLOCKING_ONLY_PRUNE_REDUCE"
  - "BLOCKING_ONLY_GREEDY_TRUE_WCD"
  - "BLOCKING_ONLY_GREEDY_PRED_WCD"
  - "BOTH_UNIFORM_EXHAUSTIVE"
  - "BOTH_UNIFORM_GREEDY_PRED_WCD"
  - "BOTH_UNIFORM_GREEDY_TRUE_WCD"

- `grid_size`: Set the maximum grid size. Choose from either 6 or 10.

- `num_instances`: Specify the number of Number of problem instances to run.

- `timeout_seconds`: Set the timeout in seconds for the experiments.

### Example

Here's an example command to run a blocking-only greedy predicted WCD experiment with a grid size of 10, 500 instances, and a timeout of 600 seconds:

```bash
python gridworld/optimal/baseline_experiments.py --experiment BLOCKING_ONLY_GREEDY_PRED_WCD --grid_size 10 --num_instances 500 --timeout_seconds 600
```

Feel free to explore different experiment types and configurations by adjusting the arguments accordingly.

### Results

After running the experiments, the results will be stored in  and `./baselines/data` directories. Check these directories for experiment-specific data files.


## WCD reduction and Time comparisons

To visualize the comparisons of performance of our approach and the baselines -- you can run the `Results Analysis.ipynb` jupyter notebook in `gridworld/optimal`.





