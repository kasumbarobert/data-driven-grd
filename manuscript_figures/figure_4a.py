"""Driver for Figure 4a (shared budget, 13×13 grid)."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manuscript_figures.helpers import PreflightRequirement, apply_label_to_pdf, run_preflight_checks

GENERATED_DIR = ROOT / "manuscript_figures" / "generated_figures"
SCRIPT_DIR = ROOT / "optimal"
FIGURE_ID = "4a"
TARGET_FILENAME = f"Figure_{FIGURE_ID}.pdf"

PYTHON = sys.executable
PLOT_MODEL_ID = "aaai25_submission_1"
PLOT_DIR = SCRIPT_DIR / "plots"
PAPER_FIGURES_DIR = PLOT_DIR / "paper_figures"
SOURCE_PDF = PLOT_DIR / "grid13" / PLOT_MODEL_ID / "wcd_reduction" / "grid13_wcd_reduction_uniform_cost.pdf"
DATA_ROOT = SCRIPT_DIR / "data"
SUMMARY_ROOT = SCRIPT_DIR / "summary_data"
PREP_SCRIPT = SCRIPT_DIR / "prepare_results_for_analysis_optimal.py"
ANALYSIS_SCRIPT = SCRIPT_DIR / "analyze_and_plot_optimal.py"
RAW_OUR_DIR = DATA_ROOT / "grid13" / "ALL_MODS_test" / "langrange_values"
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
        path=DATA_ROOT / "grid13" / "initial_true_wcd_by_id.json",
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
        path=DATA_ROOT / "grid13",
        kind="dir",
    ),
    PreflightRequirement(
        label="analysis baseline-data root",
        path=SCRIPT_DIR / "baselines" / "data" / "grid13",
        kind="dir",
    ),
]


def _has_raw_data(grid_size: int) -> bool:
    return (DATA_ROOT / f"grid{grid_size}").exists()


def _run(cmd: Sequence[str], description: str) -> None:
    try:
        subprocess.run(cmd, cwd=SCRIPT_DIR, check=True)
    except FileNotFoundError:
        print(f"[figure_4a] Skipping {description}; expected data directory missing.")


def run_generation() -> Path:
    """Regenerate the shared-budget panel and label it.""" 
    run_preflight_checks(
        figure_tag="figure_4a",
        root=ROOT,
        requirements=PRECHECK_REQUIREMENTS,
        fail_on_missing=False,
    )
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    prep_cmd = [PYTHON, "prepare_results_for_analysis_optimal.py", "--grid_size", "13"]
    plot_cmd = [
        PYTHON,
        "analyze_and_plot_optimal.py",
        "--grid_size",
        "13",
        "--time_out",
        "600",
        "--file_type",
        "pdf",
    ]
    if _has_raw_data(grid_size=13):
        print("[figure_4a] Running prepare_results_for_analysis_optimal.py (grid 13)", flush=True)
        _run(prep_cmd, description="data aggregation")
    else:
        print("[figure_4a] Raw optimal data trimmed; using precomputed summaries.")

    print("[figure_4a] Running analyze_and_plot_optimal.py (grid 13)", flush=True)
    _run(plot_cmd, description="plot generation")

    labelled_output = PAPER_FIGURES_DIR / "figure_4a.pdf"
    apply_label_to_pdf(SOURCE_PDF, labelled_output, label="a")
    return labelled_output


def main() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = run_generation()
    if not output_path.exists():
        raise FileNotFoundError(f"Expected generated figure at {output_path} but nothing was found.")

    target_path = GENERATED_DIR / TARGET_FILENAME
    shutil.copy2(output_path, target_path)
    print(f"Figure {FIGURE_ID} copied to {target_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Figure 4a.")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run preflight data checks and exit.",
    )
    args = parser.parse_args()
    if args.preflight_only:
        ok, missing = run_preflight_checks(
            figure_tag="figure_4a",
            root=ROOT,
            requirements=PRECHECK_REQUIREMENTS,
            fail_on_missing=False,
        )
        if not ok:
            formatted = "\n".join(f"  - {item}" for item in missing)
            raise SystemExit(
                "Figure 4a preflight failed. Missing inputs:\n"
                f"{formatted}"
            )
        print("[figure_4a] Preflight checks passed.")
    else:
        main()
