"""Aggregate suboptimal optimization/baseline results into summary CSV files.

This replaces notebook-exported logic with a clean CLI script while preserving
legacy output locations and filenames consumed by downstream plotting scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


DEFAULT_LAMBDA_VALUES = [
    0,
    0.0001,
    0.0002,
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    1.0,
    2,
    5,
    10,
]
DEFAULT_MAX_BUDGETS = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
BASELINE_EXPERIMENTS = [
    "BOTH_UNIFORM_GREEDY_TRUE_WCD",
    "BOTH_UNIFORM_GREEDY_PRED_WCD",
]


def write_rows(path: Path, rows: Iterable[Iterable[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def parse_int_list(raw: str) -> list[int]:
    return [int(piece.strip()) for piece in raw.split(",") if piece.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(piece.strip()) for piece in raw.split(",") if piece.strip()]


def matches_lambda_pair(pair_lambdas: list[float], allowed: set[float]) -> bool:
    if len(pair_lambdas) < 2:
        return False
    return pair_lambdas[0] in allowed and pair_lambdas[1] in allowed


def collect_uniform_our_approach(
    base_dir: Path,
    grid_size: int,
    k_param: int,
    env_ids: list[int],
    lambda_values: list[float],
    max_budgets: list[int],
    timeout: int,
    ignore_timeout_value: int,
    initial_true_wcd: dict[str, float],
) -> None:
    langrange_dir = base_dir / "data" / f"grid{grid_size}" / f"K{k_param}" / "ALL_MODS_test" / "langrange_values"
    out_dir = (
        base_dir
        / "data"
        / f"grid{grid_size}"
        / f"K{k_param}"
        / "BOTH_UNIFORM_test"
        / f"n_lambdas_{len(lambda_values)}"
    )

    allowed_lambdas = set(lambda_values)
    all_wcd_changes: list[list[float]] = []
    all_budgets_realized: list[list[list[int]]] = []
    all_times: list[list[float]] = []

    for env_id in env_ids:
        budget_buckets_realized: list[list[int]] = [[0, 0] for _ in max_budgets]
        budget_buckets_wcd_change: list[float] = [0.0] * len(max_budgets)
        budget_buckets_times: list[float] = [0.0] * len(max_budgets)

        env_path = langrange_dir / f"env_{env_id}.json"
        if env_path.exists():
            with env_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)

            for pair in payload.get("lambda_pairs", []):
                lambdas = pair.get("lambdas", [])
                if not matches_lambda_pair(lambdas, allowed_lambdas):
                    continue

                num_changes = pair.get("num_changes", [0, 0])
                wcd_change = float(pair.get("wcd_change", 0.0))
                time_taken = float(pair.get("time_taken", 0.0))

                for i, budget in enumerate(max_budgets):
                    if sum(num_changes) <= budget and wcd_change > budget_buckets_wcd_change[i]:
                        budget_buckets_wcd_change[i] = wcd_change
                        budget_buckets_realized[i] = [int(num_changes[0]), int(num_changes[1])]
                        budget_buckets_times[i] = time_taken
        else:
            budget_buckets_times = [float(timeout)] * len(max_budgets)

        if initial_true_wcd.get(str(env_id), 0) == 0:
            budget_buckets_times = [float(ignore_timeout_value)] * len(max_budgets)

        all_wcd_changes.append(budget_buckets_wcd_change)
        all_budgets_realized.append(budget_buckets_realized)
        all_times.append(budget_buckets_times)

    label = "BOTH_UNIFORM_test"
    write_rows(out_dir / f"times_{grid_size}_{label}.csv", all_times)
    write_rows(out_dir / f"wcd_change_{grid_size}_{label}.csv", all_wcd_changes)
    write_rows(out_dir / f"budgets_{grid_size}_{label}.csv", all_budgets_realized)
    write_rows(out_dir / f"max_budgets_{grid_size}_{label}.csv", [max_budgets])


def collect_baseline(
    base_dir: Path,
    grid_size: int,
    k_param: int,
    env_ids: list[int],
    timeout: int,
    ignore_timeout_value: int,
    initial_true_wcd: dict[str, float],
    experiment_label: str,
) -> None:
    out_dir = (
        base_dir
        / "baselines"
        / "data"
        / f"grid{grid_size}"
        / f"K{k_param}"
        / f"timeout_{timeout}"
        / experiment_label
    )
    env_dir = out_dir / "individual_envs"

    all_wcd_changes: list[list[float]] = []
    all_budgets_realized: list[list[list[int]]] = []
    all_times: list[list[float]] = []

    max_budget_count = 19
    for env_id in env_ids:
        budget_buckets_realized: list[list[int]] = [[0, 0] for _ in range(max_budget_count)]
        budget_buckets_wcd_change: list[float] = [0.0] * max_budget_count
        budget_buckets_times: list[float] = [0.0] * max_budget_count

        env_path = env_dir / f"env_{env_id}.json"
        if env_path.exists():
            with env_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)

            if isinstance(payload.get("max_budgets"), list) and payload.get("max_budgets"):
                max_budget_count = int(payload["max_budgets"][0])

            budget_buckets_realized = payload.get("num_changes", budget_buckets_realized)
            budget_buckets_wcd_change = payload.get("wcd_changes", budget_buckets_wcd_change)
            budget_buckets_times = payload.get("times", budget_buckets_times)
        else:
            budget_buckets_times = [float(timeout)] * max_budget_count

        if initial_true_wcd.get(str(env_id), 0) == 0:
            budget_buckets_times = [float(ignore_timeout_value)] * max_budget_count

        all_wcd_changes.append(budget_buckets_wcd_change)
        all_budgets_realized.append(budget_buckets_realized)
        all_times.append(budget_buckets_times)

    write_rows(out_dir / f"times_{grid_size}_{experiment_label}.csv", all_times)
    write_rows(out_dir / f"wcd_change_{grid_size}_{experiment_label}.csv", all_wcd_changes)
    write_rows(out_dir / f"num_changes_{grid_size}_{experiment_label}.csv", all_budgets_realized)
    write_rows(out_dir / f"budgets_{grid_size}_{experiment_label}.csv", all_budgets_realized)
    write_rows(out_dir / f"max_budgets_{grid_size}_{experiment_label}.csv", [list(range(max_budget_count))])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare suboptimal summary CSV files for analysis/plotting.")
    parser.add_argument("--grid_size", type=int, default=6)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to process.")
    parser.add_argument("--interval", type=int, default=None, help="Environment index step size.")
    parser.add_argument(
        "--max_budgets",
        type=str,
        default=",".join(str(x) for x in DEFAULT_MAX_BUDGETS),
        help="Comma-separated budget list for BOTH_UNIFORM_test.",
    )
    parser.add_argument(
        "--lambda_values",
        type=str,
        default=",".join(str(x) for x in DEFAULT_LAMBDA_VALUES),
        help="Comma-separated lambda values used for filtering lambda pairs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    interval = args.interval if args.interval is not None else (53 if args.grid_size == 10 else 1)
    default_n = 600 if args.grid_size == 10 else 2200
    n_envs = args.num_envs if args.num_envs is not None else default_n
    env_ids = list(range(0, interval * n_envs, interval))

    timeout = args.timeout
    ignore_timeout_value = 2 * timeout
    max_budgets = parse_int_list(args.max_budgets)
    lambda_values = parse_float_list(args.lambda_values)

    initial_wcd_path = base_dir / "data" / f"grid{args.grid_size}" / f"K{args.k}" / "initial_true_wcd_by_id.json"
    with initial_wcd_path.open("r", encoding="utf-8") as handle:
        initial_true_wcd = json.load(handle)

    selected_init_true_wcd = [initial_true_wcd.get(str(env_id), 0) for env_id in env_ids]
    write_rows(
        base_dir / "data" / f"grid{args.grid_size}" / f"K{args.k}" / "selected_env_init_true_wcd.csv",
        [selected_init_true_wcd],
    )

    collect_uniform_our_approach(
        base_dir=base_dir,
        grid_size=args.grid_size,
        k_param=args.k,
        env_ids=env_ids,
        lambda_values=lambda_values,
        max_budgets=max_budgets,
        timeout=timeout,
        ignore_timeout_value=ignore_timeout_value,
        initial_true_wcd=initial_true_wcd,
    )

    for baseline_label in BASELINE_EXPERIMENTS:
        collect_baseline(
            base_dir=base_dir,
            grid_size=args.grid_size,
            k_param=args.k,
            env_ids=env_ids,
            timeout=timeout,
            ignore_timeout_value=ignore_timeout_value,
            initial_true_wcd=initial_true_wcd,
            experiment_label=baseline_label,
        )

    print("[prepare_suboptimal] Aggregation complete.")


if __name__ == "__main__":
    main()
