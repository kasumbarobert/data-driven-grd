#!/usr/bin/env python3
"""Utility to convert Jupyter notebooks into organised Python scripts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import nbformat


def slugify(name: str) -> str:
    """Return a snake_case variant of the provided notebook stem."""
    slug = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_").lower()
    return slug or "notebook"


def iter_notebooks(paths: Sequence[Path]) -> Iterable[Path]:
    """Yield every notebook path discovered under the provided roots."""
    for path in paths:
        if path.is_dir():
            yield from sorted(path.rglob("*.ipynb"))
        elif path.suffix == ".ipynb" and path.exists():
            yield path
        else:
            raise FileNotFoundError(f"Unsupported path: {path}")


def _escape_magic_argument(arg: str) -> str:
    return arg.replace("\\", "\\\\").replace("\"", "\\\"")


def transform_code(source: str) -> Tuple[str, bool, bool]:
    """Convert notebook-specific syntax into standard Python."""
    uses_get_ipython = False
    uses_subprocess = False
    transformed: List[str] = []

    for raw_line in source.splitlines():
        stripped = raw_line.lstrip()
        indent = raw_line[: len(raw_line) - len(stripped)]

        if stripped.startswith("%"):
            uses_get_ipython = True
            magic_body = stripped[1:]
            if " " in magic_body:
                magic, arg = magic_body.split(" ", 1)
            else:
                magic, arg = magic_body, ""
            escaped = _escape_magic_argument(arg)
            transformed.append(
                f"{indent}get_ipython().run_line_magic(\"{magic}\", \"{escaped}\")"
            )
            continue

        if stripped.startswith("!"):
            uses_subprocess = True
            command = stripped[1:]
            transformed.append(
                f"{indent}subprocess.run({json.dumps(command)}, shell=True, check=True)"
            )
            continue

        transformed.append(raw_line.rstrip())

    text = "\n".join(transformed).rstrip()
    return text, uses_get_ipython, uses_subprocess


def convert_notebook(nb_path: Path, *, output_dir: Path | None = None, overwrite: bool = False) -> Path:
    """Convert the notebook at `nb_path` into a Python script."""
    notebook = nbformat.read(nb_path, as_version=4)
    target_dir = output_dir or nb_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{slugify(nb_path.stem)}.py"

    if target_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {target_path}")

    sections: List[str] = []
    found_imports: List[str] = []
    needs_get_ipython = False
    needs_subprocess = False

    for idx, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown":
            lines = ["# %% [markdown]"]
            for line in cell.source.splitlines():
                content = line.rstrip()
                if content:
                    lines.append(f"# {content}")
                else:
                    lines.append("#")
            sections.append("\n".join(lines).rstrip())
            continue

        if cell.cell_type != "code":
            # Skip other cell types such as raw cells.
            continue

        converted, cell_uses_ipython, cell_uses_subprocess = transform_code(cell.source)
        needs_get_ipython = needs_get_ipython or cell_uses_ipython
        needs_subprocess = needs_subprocess or cell_uses_subprocess

        # Collect simple top-level import statements for consolidation.
        body_lines: List[str] = []
        for line in converted.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") and stripped.lstrip().startswith("#!/"):
                body_lines.append(line)
                continue
            if not line.startswith((" ", "\t")) and stripped.startswith(("import ", "from ")):
                found_imports.append(stripped)
            else:
                body_lines.append(line)

        cell_block = "\n".join(body_lines).rstrip()
        if cell_block:
            header = f"# %% [code] cell {idx}"
            sections.append("\n".join([header, cell_block]))

    import_block: List[str] = []
    if needs_subprocess:
        import_block.append("import subprocess")
    if needs_get_ipython:
        import_block.append("from IPython import get_ipython")
    ordered_imports = []
    for item in found_imports:
        if item not in ordered_imports:
            ordered_imports.append(item)
    import_block.extend(ordered_imports)

    header_lines = [
        f"\"\"\"Converted from {nb_path.name}.",
        "Generated automatically by tools/notebook_to_script.py.",
        '\"\"\"',
    ]

    output_parts = ["\n".join(header_lines).rstrip()]
    if import_block:
        output_parts.append("\n".join(import_block).rstrip())
    if sections:
        output_parts.append("\n\n".join(sections))

    target_path.write_text("\n\n".join([part for part in output_parts if part]) + "\n", encoding="utf-8")
    return target_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Notebook paths or directories to convert.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing .py files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory in which to place converted scripts.",
    )
    args = parser.parse_args(argv)

    notebooks = list(iter_notebooks([Path(p) for p in args.paths]))
    if not notebooks:
        parser.error("No notebooks found in the provided paths.")

    for nb_path in notebooks:
        target = convert_notebook(nb_path, output_dir=args.output_dir, overwrite=args.overwrite)
        print(f"Converted {nb_path} -> {target}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
