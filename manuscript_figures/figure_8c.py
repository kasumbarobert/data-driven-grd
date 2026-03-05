"""Driver for Figure 8c (runtime, suboptimal behavior setting)."""

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

from manuscript_figures.helpers import PreflightRequirement, run_preflight_checks

SCRIPT_DIR = ROOT / "suboptimal"
GENERATED_DIR = ROOT / "manuscript_figures" / "generated_figures"
RAW_DATA_DIR = SCRIPT_DIR / "data" / "grid6" / "K8" / "ALL_MODS_test" / "langrange_values"
PREP_SCRIPT = SCRIPT_DIR / "prepare_results_for_analysis_suboptimal.py"
ANALYSIS_SCRIPT = SCRIPT_DIR / "analyze_and_plot_suboptimal.py"
K8_DATA_ROOT = SCRIPT_DIR / "data" / "grid6" / "K8"
INITIAL_WCD_PATH = K8_DATA_ROOT / "initial_true_wcd_by_id.json"
BASELINE_TRUE_RAW_DIR = (
    SCRIPT_DIR
    / "baselines"
    / "data"
    / "grid6"
    / "K8"
    / "timeout_600"
    / "BOTH_UNIFORM_GREEDY_TRUE_WCD"
    / "individual_envs"
)
BASELINE_PRED_RAW_DIR = (
    SCRIPT_DIR
    / "baselines"
    / "data"
    / "grid6"
    / "K8"
    / "timeout_600"
    / "BOTH_UNIFORM_GREEDY_PRED_WCD"
    / "individual_envs"
)
SOURCE_PDF = SCRIPT_DIR / "plots" / "time" / "k8_time_uniform_cost.pdf"
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
        path=RAW_DATA_DIR,
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


def _run(cmd: Sequence[str], cwd: Path, env: dict[str, str], description: str) -> None:
    try:
        subprocess.run(cmd, cwd=cwd, check=True, env=env)
    except FileNotFoundError:
        print(f"[figure_8c] Skipping {description}; expected data directory missing.")


def main() -> None:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("QT_QPA_PLATFORM", "offscreen")

    have_required_raw_data, missing_required = run_preflight_checks(
        figure_tag="figure_8c",
        root=ROOT,
        requirements=PRECHECK_REQUIREMENTS,
        fail_on_missing=False,
    )

    if have_required_raw_data:
        _run(["python", "prepare_results_for_analysis_suboptimal.py"], SCRIPT_DIR, env, "data preparation")
        _run(["python", "analyze_and_plot_suboptimal.py"], SCRIPT_DIR, env, "analysis and plotting")
    else:
        print("[figure_8c] Raw suboptimal optimisation data trimmed; using precomputed plots.")

    if not SOURCE_PDF.exists():
        if missing_required:
            formatted = "\n".join(f"  - {item}" for item in missing_required)
            raise FileNotFoundError(
                "Figure 8c could not regenerate from raw data and no precomputed plot exists.\n"
                "Populate these paths and rerun:\n"
                f"{formatted}"
            )
        raise FileNotFoundError(f"Expected plot at {SOURCE_PDF} but it was not produced.")

    target = GENERATED_DIR / "Figure_8c.pdf"
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_PDF, target)
    print(f"Figure 8c copied to {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Figure 8c.")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run preflight data checks and exit.",
    )
    args = parser.parse_args()
    if args.preflight_only:
        ok, missing = run_preflight_checks(
            figure_tag="figure_8c",
            root=ROOT,
            requirements=PRECHECK_REQUIREMENTS,
            fail_on_missing=False,
        )
        if not ok:
            formatted = "\n".join(f"  - {item}" for item in missing)
            raise SystemExit(
                "Figure 8c preflight failed. Missing inputs:\n"
                f"{formatted}"
            )
        print("[figure_8c] Preflight checks passed.")
    else:
        main()
