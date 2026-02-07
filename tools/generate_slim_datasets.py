"""Generate lightweight placeholder datasets for training/optimization demos.

These synthetic artefacts mirror the expected on-disk structure while keeping
file sizes tiny (<100 KB). They are not meant for scientific use, but allow the
codebase to run end-to-end in demonstration mode after trimming large assets.
"""

from __future__ import annotations

import json
import pickle
import random
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def _write_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _ensure_csv(path: Path, rows: Sequence[Sequence[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(",".join(f"{value:.6f}" for value in row))
            handle.write("\n")


def _random_grid_tensor(channels: int, size: int) -> torch.Tensor:
    return torch.rand(channels, size, size)


def build_optimal_samples(grid_size: int) -> None:
    base = ROOT / "optimal" / "data" / f"grid{grid_size}"
    model_training = base / "model_training"

    # Minimal dataset mimicking (state, target) tuples
    samples = []
    for _ in range(6):
        state = _random_grid_tensor(13, grid_size)
        target = torch.rand(1)
        samples.append((state, target))

    _write_pickle(model_training / f"simulated_valids_final{grid_size}_0.pkl", samples)

    # Provide a tiny initial true WCD map
    init_map = {str(i): random.randint(0, 5) for i in range(0, 60, 10)}
    _write_json(base / "initial_true_wcd_by_id.json", init_map)

    # Create a handful of langrange json files for optimisation summaries
    lang_dir = base / "ALL_MODS_test" / "langrange_values"
    for env_id in range(3):
        payload = {
            "env_id": env_id,
            "lambda_pairs": [
                {
                    "lambdas": [0.0, 0.0],
                    "wcd_change": float(random.uniform(0.1, 2.0)),
                    "num_changes": [random.randint(0, 2), random.randint(0, 2)],
                    "time_taken": float(random.uniform(0.05, 0.4)),
                },
                {
                    "lambdas": [0.1, 0.1],
                    "wcd_change": float(random.uniform(0.2, 2.5)),
                    "num_changes": [random.randint(0, 3), random.randint(0, 3)],
                    "time_taken": float(random.uniform(0.05, 0.6)),
                },
            ],
        }
        _write_json(lang_dir / f"env_{env_id}.json", payload)

    # Minimal summary CSVs for ALL_MODS_test to keep plotting paths happy
    csv_rows = [[float(b) for b in range(1, 6)]]
    _ensure_csv(base / "ALL_MODS_test" / "budgets_grid_data_placeholder.csv", csv_rows)


def build_suboptimal_samples() -> None:
    base = ROOT / "suboptimal" / "data" / "grid6" / "model_training"
    samples = []
    for _ in range(6):
        state = _random_grid_tensor(13, 6)
        target = torch.rand(1)
        samples.append((state, target))
    _write_pickle(base / "dataset_6_K8_best.pkl", samples)


def build_overcooked_samples() -> None:
    base = ROOT / "overcooked-ai" / "src" / "overcooked_ai_py" / "simulations" / "data"

    # Actor training dataset placeholder
    samples = []
    for _ in range(6):
        layout = torch.rand(4, 6, 6)
        target = torch.rand(1)
        samples.append((layout, target))
    _write_pickle(base / "dataset_6.pkl", samples)

    # Minimal optimisation results (single environment) for analysis scripts
    lang_dir = base / "grid6" / "optim_runs" / "CONSTRAINED" / "langrange_values"
    payload = {
        "env_id": 0,
        "lambdas": [
            {
                "lambdas": [0.0, 0.0],
                "num_changes": {"O+T": 1, "S+P+D": 0},
                "wcd_change": 1.2,
                "time_taken": 0.5,
            },
            {
                "lambdas": [0.1, 0.0],
                "num_changes": {"O+T": 1, "S+P+D": 1},
                "wcd_change": 1.8,
                "time_taken": 0.8,
            },
        ],
    }
    _write_json(lang_dir / "env_0.json", payload)


def build_human_validation_samples() -> None:
    base = ROOT / "human-exp-data-driven"
    data_dir = base / "validation-study"

    # Tiny CSV with a few rows of synthetic human trajectories
    rows = [
        "participant_id,treatment,gameidx2,goal,actions,reach_goal",
        "0,0,0,0,[0,1,2,3],1",
        "1,1,0,1,[1,1,2,2],1",
    ]
    csv_path = data_dir / "formal200-reachgoal.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    # Lightweight model weights placeholder (torch state dict with small tensor)
    model_path = data_dir / "model_grid6.pt"
    torch.save({"linear.weight": torch.rand(4, 144)}, model_path)


def main() -> None:
    build_optimal_samples(grid_size=6)
    build_optimal_samples(grid_size=13)
    build_suboptimal_samples()
    build_overcooked_samples()
    build_human_validation_samples()
    print("Slim datasets generated.")


if __name__ == "__main__":
    main()
