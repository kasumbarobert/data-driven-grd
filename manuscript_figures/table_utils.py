"""Utility helpers for generating manuscript tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

DATA_PATH = Path(__file__).resolve().parent / "data" / "table_metrics.json"


class TableNotFoundError(KeyError):
    """Raised when a requested table identifier is missing from the dataset."""


def load_table_spec(table_id: str) -> Dict:
    """Return the specification for ``table_id`` from the shared metrics file."""
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    try:
        return payload[table_id]
    except KeyError as exc:
        raise TableNotFoundError(f"Unknown table identifier: {table_id}") from exc


def format_value(raw: str) -> str:
    """Format a table cell for LaTeX output."""
    return raw.replace("±", "\\pm")


def render_latex_table(columns: Iterable[str], rows: Iterable[Dict[str, str]]) -> str:
    """Render ``rows`` under ``columns`` into a LaTeX tabular environment."""
    columns = list(columns)
    newline = r" \\"  # LaTeX newline
    header = " & ".join(columns) + newline

    body_lines: List[str] = []
    for row in rows:
        cells = [format_value(str(row.get(column, ""))) for column in columns]
        body_lines.append(" & ".join(cells) + newline)

    column_alignment = "l" + "c" * (len(columns) - 1)
    lines = [
        f"\\begin{{tabular}}{{{column_alignment}}}",
        "\\toprule",
        header,
        "\\midrule",
        *body_lines,
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(lines)


def write_latex_table(table_id: str, destination: Path) -> Path:
    """Load ``table_id`` and write its LaTeX tabular content to ``destination``."""
    spec = load_table_spec(table_id)
    content = render_latex_table(spec["columns"], spec["rows"])

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        handle.write("% Generated automatically by manuscript_figures.table_utils\n")
        handle.write(content)
        handle.write("\n")

    return destination
