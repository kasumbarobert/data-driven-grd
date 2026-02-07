from pathlib import Path
import os
import shutil
import subprocess
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "overcooked-ai" / "src" / "overcooked_ai_py" / "simulations"
GENERATED_DIR = ROOT / "manuscript_figures" / "generated_figures"
RAW_DATA_DIR = SCRIPT_DIR / "data" / "grid6" / "optim_runs"
ANALYSIS_DIR = SCRIPT_DIR / "analysis"
TARGET_RELATIVE = Path("plots") / "time" / "overcooked_time.pdf"


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

    if _has_raw_data():
        _run(["python", "prepare_optim_results_for_visualization.py"], SCRIPT_DIR, env, "data preparation")
        _run(["python", "analyse_visualize_results.py"], ANALYSIS_DIR, env, "analysis and plotting")
    else:
        print("[figure_8d] Raw Overcooked-AI optimisation data trimmed; using precomputed plots.")

    source = ANALYSIS_DIR / TARGET_RELATIVE
    target = GENERATED_DIR / "Figure_8d.pdf"
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(f"Figure 8d copied to {target}")


if __name__ == "__main__":
    main()
