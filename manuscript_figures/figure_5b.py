"""
Driver script for Figure 5b (Suboptimal grid-world WCD reduction).

This figure corresponds to the suboptimal-agent panel in Figure 5 of the
paper. It orchestrates the baselines, our optimization routine, and the
analysis script that produces `k8_wcd_reduction_uniform_cost.pdf`, then stages
the PDF as `generated_figures/Figure_5b.pdf`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import shutil
import subprocess
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manuscript_figures.helpers import PreflightRequirement, apply_label_to_pdf, run_preflight_checks

SUBOPTIMAL_DIR = ROOT / "suboptimal"
GENERATED_DIR = ROOT / "manuscript_figures" / "generated_figures"
FIGURE_ID = "5b"
TARGET_FILENAME = f"Figure_{FIGURE_ID}.pdf"
SOURCE_PDF = SUBOPTIMAL_DIR / "plots" / "wcd_reduction" / "k8_wcd_reduction_uniform_cost.pdf"
PREP_SCRIPT = SUBOPTIMAL_DIR / "prepare_results_for_analysis_suboptimal.py"
ANALYSIS_SCRIPT = SUBOPTIMAL_DIR / "analyze_and_plot_suboptimal.py"
K8_DATA_ROOT = SUBOPTIMAL_DIR / "data" / "grid6" / "K8"
INITIAL_WCD_PATH = K8_DATA_ROOT / "initial_true_wcd_by_id.json"
OUR_RAW_DIR = K8_DATA_ROOT / "ALL_MODS_test" / "langrange_values"
BASELINE_TRUE_RAW_DIR = (
    SUBOPTIMAL_DIR
    / "baselines"
    / "data"
    / "grid6"
    / "K8"
    / "timeout_600"
    / "BOTH_UNIFORM_GREEDY_TRUE_WCD"
    / "individual_envs"
)
BASELINE_PRED_RAW_DIR = (
    SUBOPTIMAL_DIR
    / "baselines"
    / "data"
    / "grid6"
    / "K8"
    / "timeout_600"
    / "BOTH_UNIFORM_GREEDY_PRED_WCD"
    / "individual_envs"
)
PRECHECK_REQUIREMENTS = [
    PreflightRequirement(
        label="prepare script",
        path=PREP_SCRIPT,
        kind="file",
    ),
    PreflightRequirement(
        label="analysis script",
        path=ANALYSIS_SCRIPT,
        kind="file",
    ),
    PreflightRequirement(
        label="initial true-WCD mapping",
        path=INITIAL_WCD_PATH,
        kind="file",
    ),
    PreflightRequirement(
        label="our approach raw JSONs",
        path=OUR_RAW_DIR,
        kind="glob",
        pattern="env_*.json",
    ),
    PreflightRequirement(
        label="baseline TRUE raw JSONs",
        path=BASELINE_TRUE_RAW_DIR,
        kind="glob",
        pattern="env_*.json",
    ),
    PreflightRequirement(
        label="baseline PRED raw JSONs",
        path=BASELINE_PRED_RAW_DIR,
        kind="glob",
        pattern="env_*.json",
    ),
]

PYTHON = sys.executable

# Required artefacts produced by the data-generation / training pipeline.
DATASET_PATH = SUBOPTIMAL_DIR / "data" / "grid6" / "model_training" / "dataset_6_K8_best.pkl"
MODEL_PATH = SUBOPTIMAL_DIR / "models" / "wcd_nn_model_6_K8_best.pt"


def run_step(cmd: Sequence[str], cwd: Path) -> None:
    env = {**os.environ, "MPLBACKEND": "Agg", "QT_QPA_PLATFORM": "offscreen"}
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def run_generation() -> Path:
    run_preflight_checks(
        figure_tag="figure_5b",
        root=ROOT,
        requirements=PRECHECK_REQUIREMENTS,
        fail_on_missing=True,
    )

    have_training_assets = DATASET_PATH.exists() and MODEL_PATH.exists()
    if not have_training_assets:
        print("[figure_5b] Training artefacts trimmed; skipping baseline/optimization reruns.")

    baseline_true = [
        PYTHON,
        "run_baseline_experiments_suboptimal.py",
        "--experiment_type",
        "BOTH_UNIFORM_GREEDY_TRUE_WCD",
    ]
    baseline_pred = [
        PYTHON,
        "run_baseline_experiments_suboptimal.py",
        "--experiment_type",
        "BOTH_UNIFORM_GREEDY_PRED_WCD",
    ]
    optimization_cmd = [
        PYTHON,
        "run_optimization_suboptimal.py",
        "--experiment_type",
        "BOTH_UNIFORM",
    ]
    aggregate_cmd = [PYTHON, "prepare_results_for_analysis_suboptimal.py"]
    analysis_cmd = [PYTHON, "analyze_and_plot_suboptimal.py"]

    # Analysis writes to ./plots/{time,wcd_reduction}/... and ./plot_data/...
    # Ensure those directories exist in trimmed-data checkouts.
    (SUBOPTIMAL_DIR / "plots" / "time").mkdir(parents=True, exist_ok=True)
    (SUBOPTIMAL_DIR / "plots" / "wcd_reduction").mkdir(parents=True, exist_ok=True)
    (SUBOPTIMAL_DIR / "plot_data").mkdir(parents=True, exist_ok=True)

    if have_training_assets:
        print("[figure_5b] Running baseline (true wcd)", flush=True)
        run_step(baseline_true, SUBOPTIMAL_DIR)
        print("[figure_5b] Running baseline (predicted wcd)", flush=True)
        run_step(baseline_pred, SUBOPTIMAL_DIR)
        print("[figure_5b] Running optimization pipeline", flush=True)
        run_step(optimization_cmd, SUBOPTIMAL_DIR)

    print("[figure_5b] Aggregating results", flush=True)
    run_step(aggregate_cmd, SUBOPTIMAL_DIR)
    print("[figure_5b] Generating plots", flush=True)
    run_step(analysis_cmd, SUBOPTIMAL_DIR)

    if not SOURCE_PDF.exists():
        raise FileNotFoundError(
            f"Expected plot at {SOURCE_PDF} but it was not produced."
        )

    labelled_output = SOURCE_PDF.with_name("figure_5b.pdf")
    apply_label_to_pdf(SOURCE_PDF, labelled_output, label="b")
    return labelled_output


def main() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = run_generation()

    target_path = GENERATED_DIR / TARGET_FILENAME
    shutil.copy2(output_path, target_path)
    print(f"Figure {FIGURE_ID} copied to {target_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Figure 5b.")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run preflight data checks and exit.",
    )
    args = parser.parse_args()
    if args.preflight_only:
        ok, missing = run_preflight_checks(
            figure_tag="figure_5b",
            root=ROOT,
            requirements=PRECHECK_REQUIREMENTS,
            fail_on_missing=False,
        )
        if not ok:
            formatted = "\n".join(f"  - {item}" for item in missing)
            raise SystemExit(
                "Figure 5b preflight failed. Missing inputs:\n"
                f"{formatted}"
            )
        print("[figure_5b] Preflight checks passed.")
    else:
        main()
