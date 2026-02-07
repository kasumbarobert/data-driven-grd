"""Driver for Figure 6a (human-subject overlapping actions)."""

from pathlib import Path
import os
import shutil
import subprocess
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manuscript_figures.helpers import apply_label_to_pdf

GENERATED_DIR = ROOT / "manuscript_figures" / "generated_figures"
SCRIPT_DIR = ROOT / "human-exp-data-driven" / "validation-study"
FIGURE_ID = "6a"
TARGET_FILENAME = f"Figure_{FIGURE_ID}.pdf"

PYTHON = sys.executable
PLOTS_DIR = SCRIPT_DIR / "plots"
SOURCE_PDF = PLOTS_DIR / "percentile_overlapping_actions.pdf"
DATA_DEPENDENCIES = [
    SCRIPT_DIR / "formal200-reachgoal.csv",
    SCRIPT_DIR / "model_grid6.pt",
]


def _has_full_dataset() -> bool:
    return all(path.exists() for path in DATA_DEPENDENCIES)


def _run(cmd: Sequence[str], cwd: Path) -> None:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("MPLCONFIGDIR", str(Path("/tmp/mplconfig")))
    env.setdefault("XDG_CACHE_HOME", str(Path("/tmp/xdg-cache")))
    # Keep OpenMP/BLAS single-threaded and avoid SHM failures in sandboxed runs.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("KMP_AFFINITY", "none")
    env.setdefault("KMP_INIT_AT_FORK", "FALSE")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    # Save only the source artifact needed by this figure driver.
    env.setdefault("FIGURE6A_TARGET_PLOT", SOURCE_PDF.name)
    # Ensure cache dirs exist
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def run_generation() -> Path:
    """Run the human-overlap analysis notebook export and label the output."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if not _has_full_dataset():
        missing_paths = [str(path) for path in DATA_DEPENDENCIES if not path.exists()]
        raise FileNotFoundError(
            "[figure_6a] Missing required data for full recomputation: " + ", ".join(missing_paths)
        )

    # Remove stale output so this run must regenerate from source analysis.
    if SOURCE_PDF.exists():
        SOURCE_PDF.unlink()

    cmd = [PYTHON, "perfomance_by_actual_human_distinctiveness.py"]
    print("[figure_6a] Recomputing from scratch via perfomance_by_actual_human_distinctiveness.py", flush=True)
    _run(cmd, SCRIPT_DIR)

    if not SOURCE_PDF.exists():
        raise FileNotFoundError(f"[figure_6a] Analysis completed but did not create {SOURCE_PDF}.")

    labelled_output = PLOTS_DIR / "figure_6a.pdf"
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
    main()
