"""Driver for Figure 4b (individual budgets, 13×13 grid)."""

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
FIGURE_ID = "4b"
TARGET_FILENAME = f"Figure_{FIGURE_ID}.pdf"

PYTHON = sys.executable
MODEL_ID = "aaai25-model-13"
RATIO_ID = "ratio_5_1"
PLOT_DIR = SCRIPT_DIR / "plots"
PAPER_FIGURES_DIR = PLOT_DIR / "paper_figures"
SOURCE_PDF = PLOT_DIR / "grid13" / MODEL_ID / "wcd_reduction" / f"grid13_wcd_reduction_ratio_{RATIO_ID}.pdf"
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
        print(f"[figure_4b] Skipping {description}; expected data directory missing.")


def run_generation() -> Path:
    """Regenerate the individual-budget panel and label it."""
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    prep_cmd = [PYTHON, "prepare_data_for_analysis.py", "--grid_size", "13", "--wcd_pred_model_id", MODEL_ID]
    plot_cmd = [
        PYTHON,
        "analyse_and_plot.py",
        "--grid_size",
        "13",
        "--time_out",
        "600",
        "--file_type",
        "pdf",
        "--wcd_pred_model_id",
        MODEL_ID,
    ]
    if _has_raw_data(grid_size=13):
        print("[figure_4b] Running prepare_data_for_analysis.py (grid 13)", flush=True)
        _run(prep_cmd, description="data aggregation")
    else:
        print("[figure_4b] Raw optimal data trimmed; using precomputed summaries.")

    if not _has_summary_data(grid_size=13, model_id=MODEL_ID):
        raise FileNotFoundError(
            "Expected summary statistics at "
            f"{SUMMARY_ROOT / 'grid13' / 'ml-our-approach' / MODEL_ID}."
        )

    print("[figure_4b] Running analyse_and_plot.py (grid 13)", flush=True)
    _run(plot_cmd, description="plot generation")

    labelled_output = PAPER_FIGURES_DIR / "figure_4b.pdf"
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
