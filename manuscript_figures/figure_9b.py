"""Driver for Figure 9b (6×6 grid, shared budget)."""

from pathlib import Path
import shutil
import subprocess
import sys
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manuscript_figures.helpers import apply_label_to_pdf

GENERATED_DIR = ROOT / "manuscript_figures" / "generated_figures"
SCRIPT_DIR = ROOT / "optimal"
FIGURE_ID = "9b"
TARGET_FILENAME = f"Figure_{FIGURE_ID}.pdf"

PYTHON = sys.executable
MODEL_ID = "aaai25-model-6"
PLOT_DIR = SCRIPT_DIR / "plots"
PAPER_FIGURES_DIR = PLOT_DIR / "paper_figures"
SOURCE_PDF = PLOT_DIR / "grid6" / MODEL_ID / "wcd_reduction" / "grid6_wcd_reduction_uniform_cost.pdf"
DATA_ROOT = SCRIPT_DIR / "data"
SUMMARY_ROOT = SCRIPT_DIR / "summary_data"
MODEL_ID_ALIASES = {
    "aaai25-model-6": ["aaai25_submission_1"],
    "aaai25-model-13": ["aaai25_submission_1"],
}


def _has_raw_data(grid_size: int) -> bool:
    return (DATA_ROOT / f"grid{grid_size}").exists()


def _resolve_summary_dir(grid_size: int, model_id: str) -> Optional[Path]:
    base = SUMMARY_ROOT / f"grid{grid_size}" / "ml-our-approach"
    candidates = [base / model_id]
    candidates.extend(base / alias for alias in MODEL_ID_ALIASES.get(model_id, []))
    if "_" in model_id:
        canonical_id = model_id.split("_")[0]
        candidates.append(base / canonical_id)
    for path in candidates:
        if path.exists():
            return path
    return None


def _has_summary_data(grid_size: int, model_id: str) -> bool:
    return _resolve_summary_dir(grid_size, model_id) is not None


def _run(cmd: Sequence[str], description: str) -> None:
    try:
        subprocess.run(cmd, cwd=SCRIPT_DIR, check=True)
    except FileNotFoundError:
        print(f"[figure_9b] Skipping {description}; expected data directory missing.")


def run_generation() -> Path:
    """Regenerate the 6×6 shared-budget panel and label it."""
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    prep_cmd = [PYTHON, "prepare_data_for_analysis.py", "--grid_size", "6", "--wcd_pred_model_id", MODEL_ID]
    plot_cmd = [
        PYTHON,
        "analyse_and_plot.py",
        "--grid_size",
        "6",
        "--time_out",
        "600",
        "--file_type",
        "pdf",
        "--wcd_pred_model_id",
        MODEL_ID,
    ]
    if _has_raw_data(grid_size=6):
        print("[figure_9b] Running prepare_data_for_analysis.py (grid 6)", flush=True)
        _run(prep_cmd, description="data aggregation")
    else:
        print("[figure_9b] Raw optimal data trimmed; using precomputed summaries.")

    if not _has_summary_data(grid_size=6, model_id=MODEL_ID):
        raise FileNotFoundError(
            "Expected summary statistics at "
            f"{SUMMARY_ROOT / 'grid6' / 'ml-our-approach' / MODEL_ID}."
        )

    print("[figure_9b] Running analyse_and_plot.py (grid 6)", flush=True)
    _run(plot_cmd, description="plot generation")

    labelled_output = PAPER_FIGURES_DIR / "figure_9b.pdf"
    apply_label_to_pdf(SOURCE_PDF, labelled_output, label="b")
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
