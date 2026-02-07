import sys
from pathlib import Path
# sys.path.insert(0, "./")
# sys.path.insert(0, "../../")
import torch
from torch.utils.data import Dataset
from utils import *
import random
import matplotlib.pyplot as plt

from torchvision.models import resnet50, resnet18
import argparse
import traceback
import seaborn as sns
import pandas as pd
import csv
import ast
import pdb
from scipy.interpolate import make_interp_spline
from scipy.ndimage import uniform_filter1d
import argparse
import pandas as pd
import numpy as np
import json


TRIMMED_MODEL_IDS = {"aaai25_submission", "aaai25-model-6", "aaai25-model-13"}
DEFAULT_MODEL_ID_BY_GRID = {6: "aaai25-model-6", 13: "aaai25-model-13"}
MODEL_ID_ALIASES = {
    "aaai25-model-6": ["aaai25_submission_1"],
    "aaai25-model-13": ["aaai25_submission_1"],
}


display_label = {
    "BLOCKING_ONLY_EXHAUSTIVE":"Exhaustive",
    "BLOCKING_ONLY_PRUNE_REDUCE": "Pruned-Reduce",
    "BLOCKING_ONLY_GREEDY_TRUE_WCD":"Greedy (true wcd)",
    "BLOCKING_ONLY_GREEDY_PRED_WCD":"Greedy (predicted wcd)",
    "BLOCKING_ONLY_test":"Our approach",
    "ALL_MODS_EXHAUSTIVE":"Exhaustive",
    "ALL_MODS_GREEDY_PRED_WCD":"Greedy (predicted wcd)",
    "ALL_MODS_GREEDY_TRUE_WCD":"Greedy (true wcd)",
    "ALL_MODS_test":"Our approach",
    "BOTH_UNIFORM_EXHAUSTIVE":"Exhaustive",
    "BOTH_UNIFORM_GREEDY_TRUE_WCD":"Greedy (true wcd)",
    "BOTH_UNIFORM_GREEDY_PRED_WCD":"Greedy (predicted wcd)",
    "BOTH_UNIFORM_test":"Our approach"
    
}

display_label_colors = {
    "BLOCKING_ONLY_EXHAUSTIVE": "#333333",  # A blackish color
    "BLOCKING_ONLY_PRUNE_REDUCE": "#d95f02",  # An orangish color
    "BLOCKING_ONLY_GREEDY_TRUE_WCD": "#7570b3",  # A bluish color
    "BLOCKING_ONLY_GREEDY_PRED_WCD": "#e7298a",  # A crimson-like pink color
    "BLOCKING_ONLY_test": "#66a61e",  # A grass green color
    "ALL_MODS_EXHAUSTIVE": "#333333",  # A golden color
    "ALL_MODS_GREEDY_PRED_WCD": "#e7298a",   # A crimson-like pink color
    "ALL_MODS_GREEDY_TRUE_WCD": "#7570b3",  # A bluish color
    "ALL_MODS_test": "#66a61e",  # A grass green color
    "BOTH_UNIFORM_EXHAUSTIVE": "#333333",  # A golden color
    "BOTH_UNIFORM_GREEDY_PRED_WCD": "#e7298a",   # A crimson-like pink color
    "BOTH_UNIFORM_GREEDY_TRUE_WCD": "#7570b3",  # A bluish color
    "BOTH_UNIFORM_test": "#66a61e"  # A grass green color
}

plot_markers = {
    "BLOCKING_ONLY_EXHAUSTIVE": "-*",  # A blackish color
    "BLOCKING_ONLY_PRUNE_REDUCE": "-.",  # An orangish color
    "BLOCKING_ONLY_GREEDY_TRUE_WCD": "-o",  # A bluish color
    "BLOCKING_ONLY_GREEDY_PRED_WCD": "-o",  # A crimson-like pink color
    "BLOCKING_ONLY_test": "-o",  # A grass green color
    "ALL_MODS_EXHAUSTIVE": "-o",  # A golden color
    "ALL_MODS_GREEDY_PRED_WCD": "-o",   # A crimson-like pink color
    "ALL_MODS_GREEDY_TRUE_WCD": "-o",  # A bluish color
    "ALL_MODS_test": "-o",  # A grass green color
    "BOTH_UNIFORM_EXHAUSTIVE": "-o",  # A golden color
    "BOTH_UNIFORM_GREEDY_PRED_WCD": "-o",   # A crimson-like pink color
    "BOTH_UNIFORM_GREEDY_TRUE_WCD": "-o",  # A bluish color
    "BOTH_UNIFORM_test": "-o"  # A grass green color
}




def candidate_model_ids(model_id: str) -> list[str]:
    """Return possible identifiers for locating precomputed artefacts."""

    candidates = [model_id]
    candidates.extend(MODEL_ID_ALIASES.get(model_id, []))
    if "_" in model_id:
        candidates.append(model_id.split("_")[0])
    return candidates


def resolve_summary_subdir(base: Path, model_id: str) -> Path:
    """Return the existing summary directory matching ``model_id`` if present."""

    for candidate in candidate_model_ids(model_id):
        candidate_path = base / candidate
        if candidate_path.exists():
            return candidate_path
    return base / model_id


def first_existing_path(*paths: Path) -> Path:
    """Return the first existing path from ``paths`` (or the last candidate)."""

    for path in paths:
        if path and path.exists():
            return path
    return paths[-1]


def extract_labels(folder_path, grid_size=6):
    """Return experiment labels available under ``folder_path``.

    The original notebook assumed every directory was present; here we guard
    against trimmed datasets by returning an empty list when the directory is
    missing.
    """

    folder = Path(folder_path)
    if not folder.exists():
        return []

    folder_names = []
    for entry in folder.iterdir():
        if entry.is_dir() and entry.name in display_label:
            folder_names.append(entry.name)

    return folder_names

def read_env_data(file_name):
    with open(file_name, "rb") as f:
        loaded_dataset = pickle.load(f)
        x_data = []
        y_data = []
        for i in range(loaded_dataset. __len__()):

            x_data.append(loaded_dataset[i][0].unsqueeze(0))
            y_data.append(loaded_dataset[i][1].unsqueeze(0))

        x_init_data = torch.cat(x_data).numpy()
        y_init_data = torch.cat(y_data).numpy()
        
    return x_init_data, y_init_data

def read_csv(filename):
    """
    Reads numbers from a CSV file and returns them as a list of lists.

    Args:
    filename (str): The name of the CSV file.

    Returns:
    list of lists: Each sublist contains numbers from a row in the CSV file.
    """
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Convert each string in the row to an integer
            number_row = [ast.literal_eval(item) for item in row]
            data.append(number_row)
    return data 


def sort_with_correspondence(arr_to_sort, arr_to_rearrange):
    """
    Sorts each row of 'arr_to_sort' and rearranges each corresponding row of 'arr_to_rearrange' to match.

    Parameters:
    arr_to_sort (np.array): The array whose rows are to be sorted.
    arr_to_rearrange (np.array): The array whose rows are to be rearranged to match the sorting of 'arr_to_sort'.

    Returns:
    (np.array, np.array): A tuple containing the sorted 'arr_to_sort' and the correspondingly rearranged 'arr_to_rearrange'.
    """
    sorted_arr = np.empty_like(arr_to_sort)
    rearranged_arr = np.empty_like(arr_to_rearrange)

    for i in range(arr_to_sort.shape[0]):
        # Get the indices that would sort the current row
        sorted_indices = np.argsort(arr_to_sort[i])

        # Sort the current row in 'arr_to_sort' and rearrange the corresponding row in 'arr_to_rearrange'
        sorted_arr[i] = arr_to_sort[i][sorted_indices]
        rearranged_arr[i] = arr_to_rearrange[i][sorted_indices]

    return sorted_arr, rearranged_arr

    

def moving_average(data, window_size):
    return uniform_filter1d(data, size=window_size, mode='reflect')

def plot_summary(df, ylabel, title, show_std_err=True, filename=None, use_given_budget=True, smoothing_window=4, 
                 use_log_scale=False, show_title=False, n_lambda=17, grid_size=6, time_out=600, plot_dir=f"./plots"):
    """
    Generates and saves summary plots based on the provided DataFrame and parameters.

    Parameters:
        df (DataFrame): The input data containing budget and mean values.
        ylabel (str): The label for the y-axis.
        title (str): The title for the plot.
        show_std_err (bool): Whether to display standard error bands.
        filename (str): Path to save the output plot. If None, the plot is not saved.
        use_given_budget (bool): Whether to use the given budget column or realized budget column.
        smoothing_window (int): The size of the smoothing window for moving average.
        use_log_scale (bool): Whether to apply a logarithmic scale to the mean values.
        show_title (bool): Whether to display the plot title.
        n_lambda (int): A parameter for test labels.
        grid_size (int): The size of the grid (affects budget range).
        time_out (int): The timeout threshold for data processing.
    """
    font_size = 70  # Font size for plot labels and ticks
    lw = 4  # Line width for plot lines
    max_budget = 21 if grid_size == 6 else 41  # Determine max budget based on grid size

    # Determine if we need separate subplots based on whether the ylabel contains "Time"
    is_time_plot = "Time" in ylabel
    fig, ax = plt.subplots(figsize=(20, 16), dpi=300, constrained_layout=True)  # Create the figure and axes
    axes = ax if not is_time_plot else [ax]  # Use a list of axes for time plots, single axis otherwise
    if is_time_plot:
        y_ticks = np.arange(-3, 6.5, 1) if grid_size ==6 else np.arange(-1, 3, 1)
        y_lim = -3 if grid_size ==6 else -1.5
    else:
        y_lim = 0 # min WCD is 0

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(f"{plot_dir}/plot_data", exist_ok=True)

    for label, row in df.iterrows():  # Iterate through each row in the DataFrame
        legend_label = f"{display_label[label]}"  # Create the legend label for the current data series

        # import pdb; pdb.set_trace()
        # Prepare the data for plotting
        data = pd.DataFrame({
            'budget': row['given_budget'] if use_given_budget else row['realized_budget'],
            'mean': row['mean']
        })
        data["budget"] += (data["budget"] + 1) % 2  # Adjust budgets to only include odd numbers

        # Group data by budget and compute summary statistics
        grouped_data = data.groupby('budget')['mean'].agg(['mean', 'sem', 'count']).reset_index()
        grouped_data = grouped_data[grouped_data["count"] > 1]  # Filter for budgets with sufficient data points
        grouped_data = grouped_data[grouped_data["budget"] < max_budget]  # Exclude data points beyond max budget

        if use_log_scale:  # Apply logarithmic scaling if specified
            grouped_data['sem'] = grouped_data['sem'] / grouped_data['mean']
            grouped_data["mean"] = np.log10(grouped_data["mean"])

        if smoothing_window:  # Apply smoothing to the mean values
            grouped_data['mean'] = moving_average(grouped_data['mean'], smoothing_window)

        # Timeout analysis for time plots
        if is_time_plot:
            timeout_data = data.groupby('budget')['mean'].agg(['count', lambda x: (x >= time_out).sum()]).reset_index()
            timeout_data.columns = ['budget', 'total_count', 'count_timeout']  # Rename columns
            timeout_data['percentage_timeout'] = (timeout_data['count_timeout'] / timeout_data['total_count']) * 100  # Calculate timeout percentage
            print(timeout_data[timeout_data["budget"] < 20])  # Display timeout stats for debugging

        # Plot the data
        ax = axes if not is_time_plot else axes[0]  # Use appropriate axis based on plot type
        ax.errorbar(grouped_data['budget'], grouped_data['mean'], yerr=grouped_data['sem'], 
                    fmt=plot_markers[label], capsize=1, label=legend_label, 
                    color=display_label_colors[label], linewidth=lw)  # Plot the data with error bars
        if show_std_err:  # Optionally add standard error shading
            ax.fill_between(grouped_data["budget"], grouped_data['mean'] - grouped_data['sem'],
                            grouped_data['mean'] + grouped_data['sem'], alpha=0.2,
                            color=display_label_colors[label])

        if "test" in label:  # Adjust the label for test data
            label = f"{label}_{str(n_lambda).zfill(3)}"
        grouped_data.to_csv(f"{plot_dir}/plot_data/{label}_{'time' if is_time_plot else 'wcd'}.csv")  # Save the grouped data to a CSV file

    # Configure the common plot settings
    for ax in (axes if is_time_plot else [axes]):
        ax.set_xlabel("budget", fontsize=font_size)  # Set x-axis label
        ax.set_ylabel(ylabel, fontsize=font_size)  # Set y-axis label
        ax.legend(fontsize=font_size - 5 if grid_size == 6 else font_size)  # Add legend with appropriate size
        plt.xticks(range(0, max_budget, 5 if grid_size == 6 else 10), fontsize=font_size)  # Set x-ticks
        plt.yticks(fontsize=font_size)  # Set y-ticks
        ax.tick_params(axis='both', which='major', length=font_size / 2)  # Adjust tick size
        if show_title:  # Optionally add a title to the plot
            ax.set_title(title)
        
        if is_time_plot:
            plt.yticks(y_ticks,fontsize = font_size)
            plt.ylim(y_lim)
        else:
            ax.set_ylim(y_lim)

    # Save the plot to a file if a filename is provided
    if filename:
        plt.savefig(f"{plot_dir}/{filename}", dpi=500, bbox_inches='tight')
    plt.close()  # Close the plot to free up memory


def initialize_dataframes():
    # Initialize DataFrames to store summary data for time and WCD metrics
    columns = ["Experiment Label", "n", 'mean', 'std_err', "given_budget", "realized_budget"]
    time_summary = pd.DataFrame(columns=columns)
    wcd_summary = pd.DataFrame(columns=columns)
    return time_summary, wcd_summary

def extract_experiment_labels(grid_size, time_out, our_dir, baselines_dir, ml_greedy_dir):
    """Collect the experiment labels available for the requested plots."""

    labels = set()
    labels.update(extract_labels(our_dir, grid_size=grid_size))

    baseline_timeout_dir = Path(baselines_dir) / f"timeout_{time_out}"
    labels.update(extract_labels(baseline_timeout_dir, grid_size=grid_size))

    greedy_timeout_dir = Path(ml_greedy_dir) / f"timeout_{time_out}"
    labels.update(extract_labels(greedy_timeout_dir, grid_size=grid_size))

    return sorted(labels)

def set_experiment_parameters(grid_size):
    # Set parameters like indices and ratios based on the grid size
    if grid_size == 13:
        return {
            "select_idx_blocking_only": 224, 
            "select_idx_all_mods": 224,
            "select_idx_both_uniform": 224,
            "ratio": "ratio_5_1",
            "n_lambda": 17,
        }
    else:  # For grid size 6
        return {
            "select_idx_blocking_only": 500,
            "select_idx_all_mods": 138,
            "select_idx_both_uniform": 235,
            "ratio": "ratio_3_1",
            "n_lambda": 17,
        }

def filter_labels(labels, grid_size, use_completed_only):
    # Filter out labels containing "EXHAUSTIVE" unless they have "BLOCKING_ONLY"
    if grid_size >= 10 and use_completed_only:
        return [label for label in labels if "EXHAUSTIVE" not in label or "BLOCKING_ONLY" in label]
    return labels

    

def construct_base_dir(grid_size, experiment_label, parameters, time_out, baselines_data_dir, our_approach_data_dir,ml_greedy_data_dir):
    # Construct the base directory path based on experiment label and parameters
    if experiment_label in ["BLOCKING_ONLY_test", "ALL_MODS_test", "BOTH_UNIFORM_test"]:
        base_dir = f"{our_approach_data_dir}/{experiment_label}"
        if "ALL_MODS" in experiment_label:
            base_dir += f"/{parameters['ratio']}"
        if experiment_label != "BOTH_UNIFORM_test":
            base_dir += f"/n_lambdas_{parameters['n_lambda']}"
        else:
            base_dir += f"/n_lambdas_{parameters['n_lambda']}"
    else:
        if "PRED_WCD" in experiment_label:
            base_dir = f"{ml_greedy_data_dir}/timeout_{parameters['time_out']}/{experiment_label}" #"./data"
        else:
            base_dir = f"{baselines_data_dir}/timeout_{parameters['time_out']}/{experiment_label}" #"./data"
        # base_dir = f"{baselines_data_dir}/timeout_{time_out}/{experiment_label}"
        if "ALL_MODS" in experiment_label:
            base_dir += f"/{parameters['ratio']}"
    return base_dir

def filter_times(times, selections, time_out):
    # Filter the times array to include only entries where all times are below the timeout
    return np.logical_and(selections, (times < time_out).all(axis=1))

def process_experiment_data(base_dir, grid_size, experiment_label, selections, time_out, time_summary, wcd_summary):
    # Load times, budgets, and WCD change data from CSV files
    base_dir = Path(base_dir)
    times_path = base_dir / f"times_{grid_size}_{experiment_label}.csv"
    if not times_path.exists():
        print(f"[analyse_and_plot] Skipping {experiment_label}; missing {times_path}")
        return

    times = np.array(read_csv(times_path))
    budgets = np.array(read_csv(base_dir / f"budgets_{grid_size}_{experiment_label}.csv")).flatten()
    wcd_change = np.array(read_csv(base_dir / f"wcd_change_{grid_size}_{experiment_label}.csv"))
    select_idx = len(selections)

    if experiment_label in ["ALL_MODS_test","BLOCKING_ONLY_test", "BOTH_UNIFORM_test"]:
        # if experiment_label== "ALL_MODS_test": continue
        print(experiment_label)
        budgets = read_csv(base_dir / f"num_changes_{grid_size}_{experiment_label}.csv")
        budgets = np.array(budgets)
        max_budgets = np.array(read_csv(base_dir / f"budgets_{grid_size}_{experiment_label}.csv"))[0]
        print("Budget",budgets.shape)
        wcd_change = np.array(wcd_change)[0:select_idx][selections]
        times = times[0:select_idx][selections]
        budgets = budgets[0:select_idx][selections]
        
        
        
        flattened_times = times.flatten()
        flattened_wcd_changes = wcd_change.flatten()
        flattened_realized_budgets = budgets.flatten()
        flattened_given_budgets =np.tile(max_budgets, times.shape[0])
        n = times.shape[0]
        
        print(flattened_wcd_changes.shape[0],flattened_given_budgets.shape[0],flattened_times.shape[0])
        time_summary.loc[experiment_label] = [experiment_label,n,flattened_times, flattened_times,flattened_given_budgets, flattened_realized_budgets]
        wcd_summary.loc[experiment_label] = [experiment_label,n,flattened_wcd_changes, flattened_wcd_changes,flattened_given_budgets, flattened_realized_budgets]
    else:
        print(experiment_label)
        wcd_change= wcd_change[0:select_idx][selections]# only envs that completed
        n = times.shape[0]
        #ONLY those that completed
        realized_budgets = np.array(read_csv(base_dir / f"num_changes_{grid_size}_{experiment_label}.csv")).sum(axis=2)[0:select_idx][selections]
        #.flatten()
        given_budget = np.array(read_csv(base_dir / f"budgets_{grid_size}_{experiment_label}.csv") * n)[0:select_idx][selections]
        times= times[0:select_idx][selections]
        n = times.shape[0]
        time_summary.loc[experiment_label] = [experiment_label,n,times.flatten(), times.flatten(),given_budget.flatten(),realized_budgets.flatten()]
        wcd_summary.loc[experiment_label] = [experiment_label,n,wcd_change.flatten(), wcd_change.flatten(),given_budget.flatten(),realized_budgets.flatten()]

def compute_selections(experiment_labels, grid_size, parameters, use_completed_only, baselines_data_dir, our_approach_data_dir,ml_greedy_data_dir):

    (
        selections_blocking_only,
        selections_both_uniform,
        selections_all_mods
    ) =  (
        np.array([True] * parameters["select_idx_blocking_only"]),
        [True] * parameters["select_idx_both_uniform"],
        [True] * parameters["select_idx_all_mods"]
    )

    if not use_completed_only:
        return (
                    selections_blocking_only,
                    selections_both_uniform,
                    selections_all_mods
                )
    time_out = parameters['time_out']
    n_lambda = parameters['n_lambda']
    for experiment_label in experiment_labels:
        # if "PRED_WCD" in experiment_label: continue 
        if experiment_label in ["BLOCKING_ONLY_test","ALL_MODS_test","BOTH_UNIFORM_test"]:
            base_data_dir = f"{our_approach_data_dir}/{experiment_label}" #"./data"
            if "ALL_MODS" in experiment_label:
                base_data_dir = f"{base_data_dir}/{parameters['ratio']}/"
            if experiment_label != "BOTH_UNIFORM_test":
                base_data_dir = f"{base_data_dir}/n_lambdas_{parameters['n_lambda']}"
            else:
                base_data_dir = f"{base_data_dir}/n_lambdas_{n_lambda}"
                
        else:
            if "PRED_WCD" in experiment_label:
                base_data_dir = f"{ml_greedy_data_dir}/timeout_{parameters['time_out']}/{experiment_label}" #"./data"
            else:
                base_data_dir = f"{baselines_data_dir}/timeout_{parameters['time_out']}/{experiment_label}" #"./data"
            if "ALL_MODS" in experiment_label:
                base_data_dir = f"{base_data_dir}/{parameters['ratio']}/"
        
        
        if "BLOCKING_ONLY" in experiment_label:
            select_idx = len(selections_blocking_only)
        elif "BOTH_UNIFORM" in experiment_label:
            select_idx = len(selections_both_uniform)
        else:
            select_idx = len(selections_all_mods)
            
        times_path = Path(base_data_dir) / f"times_{grid_size}_{experiment_label}.csv"
        if not times_path.exists():
            print(f"[analyse_and_plot] Skipping {experiment_label}; missing {times_path}")
            continue

        times = np.array(read_csv(times_path))
        print(experiment_label, "Before:",times.shape)
        times=times[0:select_idx]
        print(times.shape)
        if "BLOCKING_ONLY" in experiment_label:
            selections_blocking_only  = np.logical_and(selections_blocking_only, (times<time_out).all(axis = 1))
            # print((times<time_out).any(axis = 1))
        elif "BOTH_UNIFORM" in experiment_label:
            selections_both_uniform  = np.logical_and(selections_both_uniform, (times<time_out).all(axis = 1))
        else:
            print(experiment_label)
            selections_all_mods  = np.logical_and(selections_all_mods, (times<time_out).all(axis = 1))
    return selections_blocking_only,selections_both_uniform, selections_all_mods


def generate_summaries(grid_size, time_out, use_completed_only, args):
    # Initialize dataframes for storing summary statistics
    time_summary, wcd_summary = initialize_dataframes()

    # Extract and filter experiment labels
    experiment_labels = extract_experiment_labels(
        grid_size,
        time_out,
        args.our_approach_data_dir,
        args.greedy_baselines_data_dir,
        args.ml_greedy_data_dir,
    )
    experiment_labels = filter_labels(experiment_labels, grid_size, use_completed_only)

    # Set experiment-specific parameters and initialize selection arrays
    parameters = set_experiment_parameters(grid_size)
    parameters["time_out"] = time_out
    (
        selections_blocking_only,
        selections_both_uniform,
        selections_all_mods
    ) = compute_selections(experiment_labels, grid_size, parameters, use_completed_only,args.greedy_baselines_data_dir,args.our_approach_data_dir,args.ml_greedy_data_dir)


    # Iterate over each experiment label and process its data
    for experiment_label in experiment_labels:
        print(experiment_label)
        base_dir = construct_base_dir(grid_size, experiment_label, parameters, time_out,args.greedy_baselines_data_dir,args.our_approach_data_dir,args.ml_greedy_data_dir)
        if "BLOCKING_ONLY" in experiment_label:
            selections = selections_blocking_only
        elif "BOTH_UNIFORM" in experiment_label:
            selections = selections_both_uniform
        else:
            selections = selections_all_mods

        # Process the data and update summary DataFrames
        process_experiment_data(base_dir, grid_size, experiment_label, selections, time_out, time_summary, wcd_summary)

    return time_summary, wcd_summary,parameters

def compile_model_model_performance(plot_dat_path, json_file_path, wcd_pred_model_id, grid_size):
    # Read or create the JSON file
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as file:
            summary_json_file = json.load(file)
    else:
        summary_json_file = {wcd_pred_model_id: {"ml-greedy":{},"ml-our-approach":{}}, "greedy_baseline": {},"exhaustive_baseline":{}}
    
    summary_json_file[wcd_pred_model_id]={"ml-greedy":{},"ml-our-approach":{}}
    summary_json_file["greedy_baseline"] = {}

    csv_files = [
        "BLOCKING_ONLY_EXHAUSTIVE", "BLOCKING_ONLY_PRUNE_REDUCE", "BLOCKING_ONLY_GREEDY_TRUE_WCD",
        "BLOCKING_ONLY_GREEDY_PRED_WCD", "ALL_MODS_EXHAUSTIVE", "ALL_MODS_GREEDY_TRUE_WCD",
        "ALL_MODS_GREEDY_PRED_WCD", "BOTH_UNIFORM_EXHAUSTIVE", "BOTH_UNIFORM_GREEDY_PRED_WCD",
        "BOTH_UNIFORM_GREEDY_TRUE_WCD", "ALL_MODS_test_017", "BLOCKING_ONLY_test_017", "BOTH_UNIFORM_test_017"
    ]

    def extract_final_mean_sem_from_csv(csv_path):
        plot_data_df = pd.read_csv(csv_path)

        # Get the mean, SEM, and count from the data
        mean_wcds = plot_data_df["mean"].tolist()
        sem_wcds = plot_data_df["sem"].tolist()
        n = plot_data_df["count"].iloc[-1]
        final_mean_wcd = mean_wcds[-1]
        final_mean_wcd_sem = sem_wcds[-1]
        record = f"{final_mean_wcd:.1f}(se={final_mean_wcd_sem:.1f},n={n})"

        return record

    for grouping in ["BLOCKING_ONLY", "ALL_MODS", "BOTH_UNIFORM"]:
        # Filter CSV files that have this grouping in the name
        grouping_csv_files = [file for file in csv_files if grouping in file]

        for csv_file in grouping_csv_files:
            # Construct the file path and read the CSV
            wcd_csv_path = os.path.join(plot_dat_path, f"{csv_file}_wcd.csv")
            time_csv_path = os.path.join(plot_dat_path, f"{csv_file}_time.csv")
            if not os.path.exists(wcd_csv_path):
                print(f"File not found: {wcd_csv_path}")
                continue
            
            if not os.path.exists(time_csv_path):
                print(f"File not found: {time_csv_path}")
                continue

            # Update the summary JSON
            if "TRUE_WCD" in csv_file:
                summary_json_file["greedy_baseline"][f"{grouping}_final_wcd"] = extract_final_mean_sem_from_csv(wcd_csv_path)
                summary_json_file["greedy_baseline"][f"{grouping}_time_taken"] = extract_final_mean_sem_from_csv(time_csv_path)
            elif "PRED_WCD" in csv_file:
                summary_json_file[wcd_pred_model_id]["ml-greedy"][f"{grouping}_final_wcd"] = extract_final_mean_sem_from_csv(wcd_csv_path)
                summary_json_file[wcd_pred_model_id]["ml-greedy"][f"{grouping}_time_taken"] = extract_final_mean_sem_from_csv(time_csv_path)
            elif "test" in csv_file:
                summary_json_file[wcd_pred_model_id]["ml-our-approach"][f"{grouping}_final_wcd"] = extract_final_mean_sem_from_csv(wcd_csv_path)
                summary_json_file[wcd_pred_model_id]["ml-our-approach"][f"{grouping}_time_taken"] = extract_final_mean_sem_from_csv(time_csv_path)
            elif grid_size ==6 and "EXHAUSTIVE" in csv_file:
                summary_json_file["exhaustive_baseline"][f"{grouping}_final_wcd"] = extract_final_mean_sem_from_csv(wcd_csv_path)
                summary_json_file["exhaustive_baseline"][f"{grouping}_time_taken"] = extract_final_mean_sem_from_csv(time_csv_path)
        
    # Write back to the JSON file
    with open(json_file_path, 'w') as file:
        json.dump(summary_json_file, file, indent=4)

    print(f"Updated JSON file saved at: {json_file_path}")

    # Create a unified table for all models (including baseline)
    unified_table = []

    for id in summary_json_file.keys():
        if (id == "greedy_baseline") or (id =="exhaustive_baseline" and grid_size==6):
            baseline_metrics = summary_json_file.get(id, {})
            if baseline_metrics:
                baseline_entry = {"model": id, "method": id.replace("_baseline", ""), **baseline_metrics}
                unified_table.append(baseline_entry)
        else:
            for method, metrics in summary_json_file.get(id, {}).items():
                entry = {"model": id, "method": method, **metrics}
                unified_table.append(entry)

        

    # Convert unified table to a DataFrame and save it
    unified_table_path = os.path.join(os.path.dirname(json_file_path), "unified_model_performance.csv")
    unified_table_df = pd.DataFrame(unified_table)
    unified_table_df.to_csv(unified_table_path, index=False)
    print(f"Unified model performance table saved at: {unified_table_path}")


def main():
    parser = argparse.ArgumentParser(description="Process parameters for plotting summaries.")
    parser.add_argument('--grid_size', type=int, default=13, help="Grid size (e.g., 13 or 6).")
    parser.add_argument('--time_out', type=int, default=600, help="Timeout value.")
    parser.add_argument('--use_completed_only', type=bool, default=True, help="Use completed only.")
    parser.add_argument('--use_given_budget', type=bool, default=True, help="Use given budget.")
    parser.add_argument('--smoothing_window', type=int, default=2, help="Smoothing window size.")
    parser.add_argument('--show_title', type=bool, default=False, help="Show title in plots.")
    parser.add_argument('--show_std_err', type=bool, default=True, help="Show standard error in plots.")
    parser.add_argument('--file_type', type=str, default="pdf", help="Output file type (e.g., pdf, png).")
    parser.add_argument('--wcd_pred_model_id', type=str, default=None, help="WCD_pred model")

    # experiment_label ="BLOCKING_ONLY_EXHAUSTIVE" 
    # cost = 0
    args = parser.parse_args()

    if args.wcd_pred_model_id is None:
        args.wcd_pred_model_id = DEFAULT_MODEL_ID_BY_GRID.get(args.grid_size, "aaai25-model-13")

    print(f"Running plotting generation the following configurations {args}")

    data_root = Path("./data") / f"grid{args.grid_size}"
    baseline_root = Path("./baselines/data") / f"grid{args.grid_size}"
    summary_root = Path("./summary_data") / f"grid{args.grid_size}"

    our_summary_base = summary_root / "ml-our-approach"
    ml_greedy_summary_base = summary_root / "ml-greedy"
    greedy_summary_base = summary_root / "greedy"

    if "sensitivity_analysis" in args.wcd_pred_model_id:
        baselines_id = args.wcd_pred_model_id.split("_sensitivity_analysis_")[0]
    else:
        baselines_id = args.wcd_pred_model_id

    args.our_approach_data_dir = str(
        first_existing_path(
            resolve_summary_subdir(our_summary_base, args.wcd_pred_model_id),
            data_root,
        )
    )

    args.ml_greedy_data_dir = str(
        first_existing_path(
            resolve_summary_subdir(ml_greedy_summary_base, baselines_id),
            baseline_root,
        )
    )

    args.greedy_baselines_data_dir = str(
        first_existing_path(
            greedy_summary_base,
            baseline_root,
        )
    )

    args.plot_dir = f"./plots/grid{args.grid_size}/{args.wcd_pred_model_id}/"
    
    os.makedirs(args.plot_dir, exist_ok = True)
    os.makedirs(f"{args.plot_dir}/time", exist_ok = True)
    os.makedirs(f"{args.plot_dir}/wcd_reduction", exist_ok = True)

    time_summary, wcd_summary, parameters = generate_summaries(args.grid_size, args.time_out, args.use_completed_only, args)
    
    ratio = parameters["ratio"]
    n_lambda = parameters["n_lambda"]

    time_summary_uniform_df = time_summary[time_summary['Experiment Label'].str.contains("UNIFORM")]
    wcd_summary_uniform_df = wcd_summary[wcd_summary['Experiment Label'].str.contains("UNIFORM")]

    # Splitting the DataFrame based on experiment labels
    time_summary_blocking_df = time_summary[time_summary['Experiment Label'].str.contains("BLOCKING")]
    wcd_summary_blocking_df = wcd_summary[wcd_summary['Experiment Label'].str.contains("BLOCKING")]

    time_summary_all_mods_df = time_summary[time_summary['Experiment Label'].str.contains("ALL_MODS")]
    wcd_summary_all_mods_df = wcd_summary[wcd_summary['Experiment Label'].str.contains("ALL_MODS")]

    print("........... Plotting for ALL MODS ....")

    # Plotting Time
    plot_summary(time_summary_all_mods_df, 'Mean Log Time (s)', f'Ratio constrained Modifications ({ratio[-3]}:{ratio[-1]})',
                 use_given_budget=args.use_given_budget, use_log_scale=True, show_std_err=args.show_std_err,
                 show_title=args.show_title, filename=f"time/grid{args.grid_size}_time_ratio_{ratio}.{args.file_type}",
                 smoothing_window=args.smoothing_window, n_lambda=n_lambda, grid_size=args.grid_size, time_out=args.time_out, plot_dir = args.plot_dir)

    plot_summary(wcd_summary_all_mods_df, 'wcd reduction', f'Ratio constrained Modifications ({ratio[-3]}:{ratio[-1]})',
                 use_given_budget=args.use_given_budget, show_title=args.show_title, show_std_err=args.show_std_err,
                 filename=f"wcd_reduction/grid{args.grid_size}_wcd_reduction_ratio_{ratio}.{args.file_type}",
                 smoothing_window=args.smoothing_window, n_lambda=n_lambda, grid_size=args.grid_size, time_out=args.time_out, plot_dir = args.plot_dir)

    print("...........  Plotting for BLOCKING ONLY ....")
    # Plotting Time
    plot_summary(time_summary_blocking_df, 'Mean Log Time (s)', f'Blocking Only',
                 use_given_budget=args.use_given_budget, show_std_err=args.show_std_err, show_title=args.show_title,
                 filename=f"time/grid{args.grid_size}_time_blocking_only.{args.file_type}", use_log_scale=True,
                 smoothing_window=args.smoothing_window, n_lambda=n_lambda, grid_size=args.grid_size, plot_dir = args.plot_dir)
    # Plotting WCD Change
    plot_summary(wcd_summary_blocking_df, 'wcd reduction', f'Blocking only',
                 use_given_budget=args.use_given_budget, show_title=args.show_title, show_std_err=args.show_std_err,
                 filename=f"wcd_reduction/grid{args.grid_size}_wcd_reduction_blocking_only.{args.file_type}",
                 smoothing_window=args.smoothing_window, n_lambda=n_lambda, grid_size=args.grid_size, time_out=args.time_out, plot_dir = args.plot_dir)

    print("...........  Plotting for BOTH UNIFORM ....")

    plot_summary(time_summary_uniform_df, 'Mean Log Time (s)', f'Uniform Modifications',
                 use_given_budget=args.use_given_budget, show_title=args.show_title, show_std_err=args.show_std_err,
                 filename=f"time/grid{args.grid_size}_time_uniform_cost.{args.file_type}", use_log_scale=True,
                 smoothing_window=args.smoothing_window, n_lambda=n_lambda, grid_size=args.grid_size, time_out=args.time_out, plot_dir = args.plot_dir)
    # Plotting WCD Change
    plot_summary(wcd_summary_uniform_df, 'wcd reduction', f'Uniform Modifications',
                 use_given_budget=args.use_given_budget, show_title=args.show_title, show_std_err=args.show_std_err,
                 filename=f"wcd_reduction/grid{args.grid_size}_wcd_reduction_uniform_cost.{args.file_type}",
                 smoothing_window=args.smoothing_window, n_lambda=n_lambda, grid_size=args.grid_size, time_out=args.time_out, plot_dir = args.plot_dir)
    plot_dat_path = f"{args.plot_dir}/plot_data/"
    json_file_path =f"./plots/grid{args.grid_size}/summary_performance.json"

    compile_model_model_performance(plot_dat_path=plot_dat_path, json_file_path=json_file_path, wcd_pred_model_id=args.wcd_pred_model_id, grid_size=args.grid_size)

if __name__ == "__main__":
    main()
