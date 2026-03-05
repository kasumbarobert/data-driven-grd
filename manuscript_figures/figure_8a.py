"""Driver for Figure 8a (runtime, standard setup with 6x6 grid)."""

import argparse
from pathlib import Path
import os
import shutil
import subprocess
import sys
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manuscript_figures.helpers import PreflightRequirement, run_preflight_checks

SCRIPT_DIR = ROOT / "optimal"
GENERATED_DIR = ROOT / "manuscript_figures" / "generated_figures"
DATA_ROOT = SCRIPT_DIR / "data"
SUMMARY_ROOT = SCRIPT_DIR / "summary_data"
PREP_SCRIPT = SCRIPT_DIR / "prepare_results_for_analysis_optimal.py"
ANALYSIS_SCRIPT = SCRIPT_DIR / "analyze_and_plot_optimal.py"
RAW_OUR_DIR = DATA_ROOT / "grid6" / "ALL_MODS_test" / "langrange_values"
MODEL_ID = "aaai25_submission_1"
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
        path=DATA_ROOT / "grid6" / "initial_true_wcd_by_id.json",
        kind="file",
    ),
    PreflightRequirement(
        label="our approach raw JSONs",
        path=RAW_OUR_DIR,
        kind="glob",
        pattern="env_*.json",
    ),
    PreflightRequirement(
        label="analysis our-data root",
        path=DATA_ROOT / "grid6",
        kind="dir",
    ),
    PreflightRequirement(
        label="analysis baseline-data root",
        path=SCRIPT_DIR / "baselines" / "data" / "grid6",
        kind="dir",
    ),
]


def _has_raw_data(grid_size: int) -> bool:
    return (DATA_ROOT / f"grid{grid_size}").exists()


def _run(cmd: Sequence[str], cwd: Path, env: dict[str, str], description: str) -> None:
    try:
        subprocess.run(cmd, cwd=cwd, check=True, env=env)
    except FileNotFoundError:
        print(f"[figure_8a] Skipping {description}; expected data directory missing.")


def main() -> None:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    run_preflight_checks(
        figure_tag="figure_8a",
        root=ROOT,
        requirements=PRECHECK_REQUIREMENTS,
        fail_on_missing=False,
    )

    if _has_raw_data(grid_size=6):
        _run(
            [
                "python",
                "prepare_results_for_analysis_optimal.py",
                "--grid_size",
                "6",
            ],
            cwd=SCRIPT_DIR,
            env=env,
            description="data aggregation",
        )
    else:
        print("[figure_8a] Raw optimal data trimmed; using precomputed summaries.")

    _run(
        [
            "python",
            "analyze_and_plot_optimal.py",
            "--grid_size",
            "6",
            "--time_out",
            "600",
            "--file_type",
            "pdf",
        ],
        cwd=SCRIPT_DIR,
        env=env,
        description="plot generation",
    )

    source = SCRIPT_DIR / "plots" / "grid6" / MODEL_ID / "time" / "grid6_time_blocking_only.pdf"
    target = GENERATED_DIR / "Figure_8a.pdf"
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(f"Figure 8a copied to {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Figure 8a.")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run preflight data checks and exit.",
    )
    args = parser.parse_args()
    if args.preflight_only:
        ok, missing = run_preflight_checks(
            figure_tag="figure_8a",
            root=ROOT,
            requirements=PRECHECK_REQUIREMENTS,
            fail_on_missing=False,
        )
        if not ok:
            formatted = "\n".join(f"  - {item}" for item in missing)
            raise SystemExit(
                "Figure 8a preflight failed. Missing inputs:\n"
                f"{formatted}"
            )
        print("[figure_8a] Preflight checks passed.")
    else:
        main()
