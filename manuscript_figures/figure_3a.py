"""Driver for Figure 3a (6×6 blocking-only WCD reduction)."""

from pathlib import Path
import os
import shutil
import subprocess
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "optimal"
GENERATED_DIR = ROOT / "manuscript_figures" / "generated_figures"
DATA_ROOT = SCRIPT_DIR / "data"
SUMMARY_ROOT = SCRIPT_DIR / "summary_data"
MODEL_ID = "aaai25-model-6"
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


def _run(cmd: Sequence[str], cwd: Path, env: dict[str, str], description: str) -> None:
    try:
        subprocess.run(cmd, cwd=cwd, check=True, env=env)
    except FileNotFoundError:
        print(f"[figure_3a] Skipping {description}; expected data directory missing.")


def main() -> None:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("QT_QPA_PLATFORM", "offscreen")

    if _has_raw_data(grid_size=6):
        _run([
            "python",
            "prepare_data_for_analysis.py",
            "--grid_size",
            "6",
            "--wcd_pred_model_id",
            MODEL_ID,
        ], cwd=SCRIPT_DIR, env=env, description="data aggregation")
    else:
        print("[figure_3a] Raw optimal data trimmed; using precomputed summaries.")

    if not _has_summary_data(grid_size=6, model_id=MODEL_ID):
        raise FileNotFoundError(
            "Expected summary statistics at "
            f"{SUMMARY_ROOT / 'grid6' / 'ml-our-approach' / MODEL_ID}."
        )

    _run([
        "python",
        "analyse_and_plot.py",
        "--grid_size",
        "6",
        "--time_out",
        "600",
        "--file_type",
        "pdf",
        "--wcd_pred_model_id",
        MODEL_ID,
    ], cwd=SCRIPT_DIR, env=env, description="plot generation")

    source = SCRIPT_DIR / "plots" / "grid6" / MODEL_ID / "wcd_reduction" / "grid6_wcd_reduction_blocking_only.pdf"
    target = GENERATED_DIR / "Figure_3a.pdf"
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(f"Figure 3a copied to {target}")


if __name__ == "__main__":
    main()
