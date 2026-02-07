"""Generator for Table 02 (grid 13 ablation study)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manuscript_figures.table_utils import write_latex_table

GENERATED_DIR = ROOT / "manuscript_figures" / "generated_figures"
TABLE_ID = "table_02"
TARGET_FILENAME = "Table_02.tex"


def run_generation() -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    destination = GENERATED_DIR / TARGET_FILENAME
    write_latex_table(TABLE_ID, destination)
    return destination


def main() -> None:
    output_path = run_generation()
    print(f"Table 02 written to {output_path}")


if __name__ == "__main__":
    main()
