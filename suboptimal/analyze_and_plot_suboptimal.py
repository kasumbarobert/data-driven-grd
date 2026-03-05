"""Analysis and plotting for suboptimal experiments.

This script keeps the default behavior expected by manuscript figure drivers
while replacing notebook-style globals with a CLI entry point.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.stats import mannwhitneyu, ttest_ind


DISPLAY_LABEL = {
    "BLOCKING_ONLY_EXHAUSTIVE": "Exhaustive",
    "BLOCKING_ONLY_PRUNE_REDUCE": "Pruned-Reduce",
    "BLOCKING_ONLY_GREEDY_TRUE_WCD": "Greedy (true wcd)",
    "BLOCKING_ONLY_GREEDY_PRED_WCD": "Greedy (predicted wcd)",
    "BLOCKING_ONLY_test": "Our approach",
    "ALL_MODS_EXHAUSTIVE": "Exhaustive",
    "ALL_MODS_GREEDY_PRED_WCD": "Greedy (predicted wcd)",
    "ALL_MODS_GREEDY_TRUE_WCD": "Greedy (true wcd)",
    "ALL_MODS_test": "Our approach",
    "BOTH_UNIFORM_EXHAUSTIVE": "Exhaustive",
    "BOTH_UNIFORM_GREEDY_TRUE_WCD": "Greedy (true wcd)",
    "BOTH_UNIFORM_GREEDY_PRED_WCD": "Greedy (predicted wcd)",
    "BOTH_UNIFORM_test": "Our approach",
}

DISPLAY_LABEL_COLORS = {
    "BLOCKING_ONLY_EXHAUSTIVE": "#1b9e77",
    "BLOCKING_ONLY_PRUNE_REDUCE": "#d95f02",
    "BLOCKING_ONLY_GREEDY_TRUE_WCD": "#7570b3",
    "BLOCKING_ONLY_GREEDY_PRED_WCD": "#e7298a",
    "BLOCKING_ONLY_test": "#66a61e",
    "ALL_MODS_EXHAUSTIVE": "#1b9e77",
    "ALL_MODS_GREEDY_PRED_WCD": "#e7298a",
    "ALL_MODS_GREEDY_TRUE_WCD": "#7570b3",
    "ALL_MODS_test": "#66a61e",
    "BOTH_UNIFORM_EXHAUSTIVE": "#1b9e77",
    "BOTH_UNIFORM_GREEDY_PRED_WCD": "#e7298a",
    "BOTH_UNIFORM_GREEDY_TRUE_WCD": "#7570b3",
    "BOTH_UNIFORM_test": "#66a61e",
}

DEFAULT_UNIFORM_LABELS = [
    "BOTH_UNIFORM_GREEDY_PRED_WCD",
    "BOTH_UNIFORM_GREEDY_TRUE_WCD",
    "BOTH_UNIFORM_test",
]


@dataclass
class ExperimentData:
    label: str
    times: np.ndarray
    wcd_change: np.ndarray
    given_budget: np.ndarray
    realized_budget: np.ndarray


def read_literal_csv(path: Path) -> list[list[object]]:
    rows: list[list[object]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            rows.append([ast.literal_eval(item) for item in row])
    return rows


def ensure_2d(array: np.ndarray) -> np.ndarray:
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array.reshape(1, -1)
    return array


def trim_to_shape(array: np.ndarray, rows: int, cols: int, fill_value: float = 0.0) -> np.ndarray:
    if rows == 0 or cols == 0:
        return np.empty((rows, cols))
    out = np.full((rows, cols), fill_value, dtype=float)
    src_rows = min(rows, array.shape[0])
    src_cols = min(cols, array.shape[1])
    out[:src_rows, :src_cols] = array[:src_rows, :src_cols]
    return out


def derive_given_budget(base_dir: Path, grid_size: int, label: str, rows: int, cols: int) -> np.ndarray:
    max_budget_path = base_dir / f"max_budgets_{grid_size}_{label}.csv"
    if max_budget_path.exists():
        max_budget_raw = np.asarray(read_literal_csv(max_budget_path), dtype=float)
        max_budget_raw = ensure_2d(max_budget_raw)
        budget_vector = max_budget_raw[0]
    else:
        budget_vector = np.arange(1, cols + 1, dtype=float)

    if budget_vector.size < cols:
        pad_value = budget_vector[-1] if budget_vector.size else 0.0
        budget_vector = np.pad(budget_vector, (0, cols - budget_vector.size), constant_values=pad_value)
    budget_vector = budget_vector[:cols]
    return np.tile(budget_vector, (rows, 1))


def normalize_budget_indexing(label: str, given_budget: np.ndarray) -> np.ndarray:
    """Normalize baseline budget indexing to start at 1 when serialized as 0..N-1."""
    if label.endswith("_test"):
        return given_budget
    if given_budget.size == 0:
        return given_budget
    if np.nanmin(given_budget) == 0:
        return given_budget + 1
    return given_budget


def derive_realized_budget(base_dir: Path, grid_size: int, label: str, rows: int, cols: int) -> np.ndarray:
    candidate_paths = [
        base_dir / f"num_changes_{grid_size}_{label}.csv",
        base_dir / f"budgets_{grid_size}_{label}.csv",
    ]
    for path in candidate_paths:
        if not path.exists():
            continue

        raw = np.asarray(read_literal_csv(path), dtype=float)
        if raw.ndim == 3:
            realized = raw.sum(axis=2)
        else:
            realized = ensure_2d(raw)
        return trim_to_shape(realized, rows=rows, cols=cols)

    return np.zeros((rows, cols), dtype=float)


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1 or data.size == 0:
        return data
    return uniform_filter1d(data, size=window_size, mode="reflect")


def build_data_dir(args: argparse.Namespace, label: str) -> Path:
    if label.endswith("_test"):
        base = Path(args.our_data_root) / f"grid{args.grid_size}" / f"K{args.k}" / label
        if "ALL_MODS" in label:
            return base / args.ratio / f"n_lambdas_{args.n_lambdas}"
        if "BLOCKING_ONLY" in label:
            return base / f"n_lambdas_{args.n_lambdas}"
        return base / f"n_lambdas_{int(round(math.sqrt(args.n_lambdas)))}"

    base = (
        Path(args.baseline_data_root)
        / f"grid{args.grid_size}"
        / f"K{args.k}"
        / f"timeout_{args.time_out}"
        / label
    )
    if "ALL_MODS" in label:
        return base / args.ratio
    return base


def load_experiment_data(args: argparse.Namespace, label: str) -> ExperimentData | None:
    base_dir = build_data_dir(args, label)
    times_path = base_dir / f"times_{args.grid_size}_{label}.csv"
    wcd_path = base_dir / f"wcd_change_{args.grid_size}_{label}.csv"

    if not times_path.exists() or not wcd_path.exists():
        print(f"[analyze_suboptimal] Skipping {label}; missing summary CSV(s) in {base_dir}")
        return None

    times = ensure_2d(np.asarray(read_literal_csv(times_path), dtype=float))
    wcd_change = ensure_2d(np.asarray(read_literal_csv(wcd_path), dtype=float))
    rows = min(times.shape[0], wcd_change.shape[0])
    cols = min(times.shape[1], wcd_change.shape[1])
    if rows == 0 or cols == 0:
        print(f"[analyze_suboptimal] Skipping {label}; empty times/WCD arrays")
        return None

    times = times[:rows, :cols]
    wcd_change = wcd_change[:rows, :cols]
    given_budget = derive_given_budget(base_dir, args.grid_size, label, rows, cols)
    given_budget = normalize_budget_indexing(label, given_budget)
    realized_budget = derive_realized_budget(base_dir, args.grid_size, label, rows, cols)

    return ExperimentData(
        label=label,
        times=times,
        wcd_change=wcd_change,
        given_budget=given_budget,
        realized_budget=realized_budget,
    )


def apply_completed_filter(data_items: list[ExperimentData], time_out: int, use_completed_only: bool) -> list[ExperimentData]:
    if not data_items:
        return []

    common_rows = min(item.times.shape[0] for item in data_items)
    trimmed = [
        ExperimentData(
            label=item.label,
            times=item.times[:common_rows],
            wcd_change=item.wcd_change[:common_rows],
            given_budget=item.given_budget[:common_rows],
            realized_budget=item.realized_budget[:common_rows],
        )
        for item in data_items
    ]

    if not use_completed_only:
        return trimmed

    mask = np.ones(common_rows, dtype=bool)
    for item in trimmed:
        mask &= (item.times < time_out).all(axis=1)

    return [
        ExperimentData(
            label=item.label,
            times=item.times[mask],
            wcd_change=item.wcd_change[mask],
            given_budget=item.given_budget[mask],
            realized_budget=item.realized_budget[mask],
        )
        for item in trimmed
    ]


def grouped_summary(
    budgets: np.ndarray,
    values: np.ndarray,
    use_log_scale: bool,
    smoothing_window: int,
    max_budget_plot: int,
) -> pd.DataFrame:
    plot_df = pd.DataFrame({"budget": budgets.flatten(), "value": values.flatten()})
    grouped = plot_df.groupby("budget")["value"].agg(["mean", "sem", "std", "count"]).reset_index()
    grouped = grouped[grouped["count"] > 1]
    grouped = grouped[grouped["budget"] < max_budget_plot]

    if grouped.empty:
        return grouped

    if use_log_scale:
        grouped = grouped[grouped["mean"] > 0]
        if grouped.empty:
            return grouped
        grouped["sem"] = grouped["sem"] / grouped["mean"]
        grouped["mean"] = np.log10(grouped["mean"])

    grouped["mean"] = moving_average(grouped["mean"].to_numpy(), smoothing_window)
    return grouped


def save_plot(
    data_items: Iterable[ExperimentData],
    metric: str,
    ylabel: str,
    output_file: Path,
    plot_data_dir: Path,
    n_lambdas: int,
    use_given_budget: bool,
    use_log_scale: bool,
    smoothing_window: int,
    show_std_err: bool,
    show_title: bool,
    title: str,
    grid_size: int,
) -> None:
    fig, ax = plt.subplots(figsize=(20, 16), dpi=300, constrained_layout=True)
    plot_data_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    font_size = 70
    line_width = 4
    max_budget_plot = 20 if grid_size == 6 else 40
    plotted_any = False

    for item in data_items:
        budgets = item.given_budget if use_given_budget else item.realized_budget
        values = item.times if metric == "time" else item.wcd_change

        grouped = grouped_summary(
            budgets=budgets,
            values=values,
            use_log_scale=use_log_scale,
            smoothing_window=smoothing_window,
            max_budget_plot=max_budget_plot,
        )
        if grouped.empty:
            continue

        plotted_any = True
        label_name = DISPLAY_LABEL.get(item.label, item.label)
        color = DISPLAY_LABEL_COLORS.get(item.label, "#333333")

        ax.errorbar(
            grouped["budget"],
            grouped["mean"],
            yerr=grouped["sem"],
            fmt="-o",
            capsize=1,
            linewidth=line_width,
            label=label_name,
            color=color,
        )
        if show_std_err:
            ax.fill_between(
                grouped["budget"],
                grouped["mean"] - grouped["sem"],
                grouped["mean"] + grouped["sem"],
                alpha=0.2,
                color=color,
            )

        csv_label = item.label
        if "test" in csv_label:
            csv_label = f"{csv_label}_{str(n_lambdas).zfill(3)}"
        grouped.to_csv(plot_data_dir / f"{csv_label}.csv", index=False)

    ax.set_xlabel("budget", fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.tick_params(axis="both", which="major", length=font_size / 2)
    ax.set_xticks(range(0, max_budget_plot + 1, 5 if grid_size == 6 else 10))
    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    if show_title:
        ax.set_title(title)
    if plotted_any:
        ax.legend(fontsize=font_size)
    else:
        ax.text(
            0.5,
            0.5,
            "No plottable data found",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=24,
        )

    plt.savefig(output_file, dpi=500, bbox_inches="tight")
    plt.close(fig)


def print_uniform_significance(data_items: list[ExperimentData]) -> None:
    lookup = {item.label: item.wcd_change[:, -1] for item in data_items if item.wcd_change.size > 0}
    required = ["BOTH_UNIFORM_GREEDY_PRED_WCD", "BOTH_UNIFORM_GREEDY_TRUE_WCD", "BOTH_UNIFORM_test"]
    if not all(label in lookup and lookup[label].size > 1 for label in required):
        return

    pred_wcd = lookup["BOTH_UNIFORM_GREEDY_PRED_WCD"]
    true_wcd = lookup["BOTH_UNIFORM_GREEDY_TRUE_WCD"]
    test_wcd = lookup["BOTH_UNIFORM_test"]

    t_test_pred_true = ttest_ind(pred_wcd, true_wcd)
    t_test_pred_test = ttest_ind(pred_wcd, test_wcd)
    t_test_true_test = ttest_ind(true_wcd, test_wcd)

    mannwhitney_pred_true = mannwhitneyu(pred_wcd, true_wcd)
    mannwhitney_pred_test = mannwhitneyu(pred_wcd, test_wcd)
    mannwhitney_true_test = mannwhitneyu(true_wcd, test_wcd)

    print("T-test p-values:")
    print(f"Pred WCD vs. True WCD: {t_test_pred_true.pvalue}")
    print(f"Pred WCD vs. Test WCD: {t_test_pred_test.pvalue}")
    print(f"True WCD vs. Test WCD: {t_test_true_test.pvalue}")
    print("\nMann-Whitney U test p-values:")
    print(f"Pred WCD vs. True WCD: {mannwhitney_pred_true.pvalue}")
    print(f"Pred WCD vs. Test WCD: {mannwhitney_pred_test.pvalue}")
    print(f"True WCD vs. Test WCD: {mannwhitney_true_test.pvalue}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze and plot suboptimal experiment summaries.")
    parser.add_argument("--grid_size", type=int, default=6)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--time_out", type=int, default=600)
    parser.add_argument("--ratio", type=str, default="ratio_1_3")
    parser.add_argument("--n_lambdas", type=int, default=17 * 17)
    parser.add_argument("--file_type", type=str, default="pdf")
    parser.add_argument("--smoothing_window", type=int, default=2)
    parser.add_argument("--use_completed_only", dest="use_completed_only", action="store_true")
    parser.add_argument("--include_timeouts", dest="use_completed_only", action="store_false")
    parser.set_defaults(use_completed_only=True)
    parser.add_argument("--use_given_budget", dest="use_given_budget", action="store_true")
    parser.add_argument("--use_realized_budget", dest="use_given_budget", action="store_false")
    parser.set_defaults(use_given_budget=True)
    parser.add_argument("--show_title", action="store_true", default=False)
    parser.add_argument("--hide_std_err", dest="show_std_err", action="store_false")
    parser.set_defaults(show_std_err=True)
    parser.add_argument("--our_data_root", type=str, default="./data")
    parser.add_argument("--baseline_data_root", type=str, default="./baselines/data")
    parser.add_argument("--plot_root", type=str, default="./plots")
    parser.add_argument("--plot_data_root", type=str, default="./plot_data")
    parser.add_argument(
        "--experiment_labels",
        nargs="*",
        default=DEFAULT_UNIFORM_LABELS,
        help="Experiment labels to include. Default reproduces Figure 5b.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = [label for label in args.experiment_labels if label in DISPLAY_LABEL]
    if not labels:
        raise ValueError("No valid experiment labels were provided.")

    loaded = []
    for label in labels:
        item = load_experiment_data(args, label)
        if item is not None:
            loaded.append(item)

    if not loaded:
        raise FileNotFoundError("No experiment summary files were found for the requested labels.")

    filtered = apply_completed_filter(
        loaded,
        time_out=args.time_out,
        use_completed_only=args.use_completed_only,
    )

    plot_root = Path(args.plot_root)
    plot_data_dir = Path(args.plot_data_root)

    save_plot(
        data_items=filtered,
        metric="time",
        ylabel="Mean Log Time (s)",
        output_file=plot_root / "time" / f"k{args.k}_time_uniform_cost.{args.file_type}",
        plot_data_dir=plot_data_dir,
        n_lambdas=args.n_lambdas,
        use_given_budget=args.use_given_budget,
        use_log_scale=True,
        smoothing_window=args.smoothing_window,
        show_std_err=args.show_std_err,
        show_title=args.show_title,
        title="Uniform Modifications",
        grid_size=args.grid_size,
    )

    save_plot(
        data_items=filtered,
        metric="wcd",
        ylabel="wcd reduction",
        output_file=plot_root / "wcd_reduction" / f"k{args.k}_wcd_reduction_uniform_cost.{args.file_type}",
        plot_data_dir=plot_data_dir,
        n_lambdas=args.n_lambdas,
        use_given_budget=args.use_given_budget,
        use_log_scale=False,
        smoothing_window=args.smoothing_window,
        show_std_err=args.show_std_err,
        show_title=args.show_title,
        title="Uniform Modifications",
        grid_size=args.grid_size,
    )

    print_uniform_significance(filtered)
    print("[analyze_suboptimal] Analysis complete.")


if __name__ == "__main__":
    main()
