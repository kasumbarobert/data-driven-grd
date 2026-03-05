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

SCRIPT_DIR = ROOT / "overcooked-ai" / "src" / "overcooked_ai_py" / "simulations"
GENERATED_DIR = ROOT / "manuscript_figures" / "generated_figures"
RAW_DATA_DIR = SCRIPT_DIR / "data" / "grid6" / "optim_runs"
ANALYSIS_DIR = SCRIPT_DIR / "analysis"
PREP_SCRIPT = SCRIPT_DIR / "prepare_results_for_analysis_overcooked.py"
ANALYSIS_SCRIPT = ANALYSIS_DIR / "analyze_and_plot_overcooked.py"
TARGET_RELATIVE = Path("plots") / "time" / "overcooked_time.pdf"
BASELINE_RAW_DIR = SCRIPT_DIR / "baselines" / "data" / "grid6" / "optim_runs" / "timeout_18000"
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
        label="our approach raw JSONs",
        path=RAW_DATA_DIR / "CONSTRAINED" / "langrange_values",
        kind="glob",
        pattern="env_*.json",
    ),
    PreflightRequirement(
        label="baseline raw JSONs",
        path=BASELINE_RAW_DIR,
        kind="glob",
        pattern="env_*.json",
        recursive=True,
    ),
]


def _has_raw_data() -> bool:
    constrained_dir = RAW_DATA_DIR / "CONSTRAINED" / "langrange_values"
    return constrained_dir.exists() and any(constrained_dir.iterdir())


def _run(cmd: Sequence[str], cwd: Path, env: dict[str, str], description: str) -> None:
    try:
        subprocess.run(cmd, cwd=cwd, check=True, env=env)
    except FileNotFoundError:
        print(f"[figure_8d] Skipping {description}; expected data directory missing.")


def main() -> None:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    run_preflight_checks(
        figure_tag="figure_8d",
        root=ROOT,
        requirements=PRECHECK_REQUIREMENTS,
        fail_on_missing=False,
    )

    if _has_raw_data():
        _run(["python", "prepare_results_for_analysis_overcooked.py"], SCRIPT_DIR, env, "data preparation")
        _run(["python", "analyze_and_plot_overcooked.py"], ANALYSIS_DIR, env, "analysis and plotting")
    else:
        print("[figure_8d] Raw Overcooked-AI optimisation data trimmed; using precomputed plots.")

    source = ANALYSIS_DIR / TARGET_RELATIVE
    target = GENERATED_DIR / "Figure_8d.pdf"
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(f"Figure 8d copied to {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Figure 8d.")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run preflight data checks and exit.",
    )
    args = parser.parse_args()
    if args.preflight_only:
        ok, missing = run_preflight_checks(
            figure_tag="figure_8d",
            root=ROOT,
            requirements=PRECHECK_REQUIREMENTS,
            fail_on_missing=False,
        )
        if not ok:
            formatted = "\n".join(f"  - {item}" for item in missing)
            raise SystemExit(
                "Figure 8d preflight failed. Missing inputs:\n"
                f"{formatted}"
            )
        print("[figure_8d] Preflight checks passed.")
    else:
        main()
