"""Utility helpers shared across manuscript figure drivers."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PreflightRequirement:
    """Declarative requirement used by figure preflight checks."""

    label: str
    path: Path
    kind: str = "exists"
    pattern: Optional[str] = None
    recursive: bool = False
    min_matches: int = 1
    require_readable: bool = True


def _is_readable_file(path: Path) -> bool:
    """Return True when file exists and at least one byte can be read."""

    if not path.exists() or not path.is_file():
        return False
    if not path.stat().st_size:
        return False
    try:
        with path.open("rb") as handle:
            return bool(handle.read(1))
    except OSError:
        return False


def _format_relative(path: Path, root: Path) -> str:
    """Format a path relative to root when possible."""

    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _iter_matches(path: Path, pattern: str, recursive: bool) -> Iterable[Path]:
    """Yield matching paths from `path` using optional recursive globbing."""

    if recursive:
        return path.rglob(pattern)
    return path.glob(pattern)


def _evaluate_requirement(requirement: PreflightRequirement) -> tuple[bool, str]:
    """Evaluate a single preflight requirement."""

    path = requirement.path
    kind = requirement.kind

    if kind == "exists":
        return path.exists(), "path exists"

    if kind == "file":
        if not path.exists() or not path.is_file():
            return False, "file missing"
        if requirement.require_readable and not _is_readable_file(path):
            return False, "file unreadable or empty"
        return True, "file present"

    if kind == "dir":
        return path.exists() and path.is_dir(), "directory present"

    if kind == "nonempty_dir":
        if not path.exists() or not path.is_dir():
            return False, "directory missing"
        try:
            has_entries = any(path.iterdir())
        except OSError:
            return False, "directory unreadable"
        return has_entries, "directory has entries"

    if kind == "glob":
        if not path.exists() or not path.is_dir():
            return False, "directory missing"
        pattern = requirement.pattern or "*"
        matches = []
        for candidate in _iter_matches(path, pattern, requirement.recursive):
            if not candidate.is_file():
                continue
            if requirement.require_readable and not _is_readable_file(candidate):
                continue
            matches.append(candidate)
        ok = len(matches) >= requirement.min_matches
        detail = f"{len(matches)} matching file(s) for pattern '{pattern}'"
        return ok, detail

    raise ValueError(f"Unsupported requirement kind: {kind}")


def run_preflight_checks(
    *,
    figure_tag: str,
    root: Path,
    requirements: Sequence[PreflightRequirement],
    fail_on_missing: bool = False,
    heading: str = "Preflight data checks",
) -> tuple[bool, list[str]]:
    """Print preflight status for all requirements and optionally fail on missing."""

    print(f"[{figure_tag}] {heading}")
    missing_paths: list[str] = []
    for requirement in requirements:
        ok, detail = _evaluate_requirement(requirement)
        symbol = "[x]" if ok else "[ ]"
        relative_path = _format_relative(requirement.path, root)
        print(f"[{figure_tag}]   {symbol} {requirement.label}: {relative_path} ({detail})")
        if not ok:
            missing_paths.append(relative_path)

    if missing_paths and fail_on_missing:
        formatted = "\n".join(f"  - {item}" for item in missing_paths)
        raise FileNotFoundError(
            f"{figure_tag}: missing required inputs.\n"
            "Populate these paths and rerun:\n"
            f"{formatted}"
        )

    if not missing_paths:
        print(f"[{figure_tag}]   [x] All listed requirements satisfied.")
    return (not missing_paths), missing_paths


def _create_label_overlay(
    text: str,
    page_width: float,
    page_height: float,
    x_offset: float,
    y_offset: float,
    font_size: int,
) -> "PageObject":
    """Return a transparent PDF page containing a single label."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PyPDF2 import PdfReader

    buffer = BytesIO()
    fig = plt.figure(figsize=(page_width / 72, page_height / 72))
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        x_offset / page_width,
        1 - (y_offset / page_height),
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=font_size,
        fontweight="bold",
    )
    fig.savefig(buffer, format="pdf", transparent=True)
    plt.close(fig)
    buffer.seek(0)
    return PdfReader(buffer).pages[0]


def combine_two_panel_pdfs(
    left: Path,
    right: Path,
    destination: Path,
    labels: Optional[Sequence[Optional[str]]] = None,
    label_offset: Tuple[float, float] = (36.0, 36.0),
    label_font_size: int = 48,
) -> None:
    """Place two single-page PDFs side-by-side and optionally add panel labels."""
    from PyPDF2 import PdfReader, PdfWriter

    for pdf_path in (left, right):
        if not pdf_path.exists():
            raise FileNotFoundError(f"Expected source PDF at {pdf_path}.")

    left_reader = PdfReader(str(left))
    right_reader = PdfReader(str(right))
    if not left_reader.pages or not right_reader.pages:
        raise ValueError("Input PDFs must each contain at least one page.")

    left_page = left_reader.pages[0]
    right_page = right_reader.pages[0]

    left_width = float(left_page.mediabox.width)
    right_width = float(right_page.mediabox.width)
    left_height = float(left_page.mediabox.height)
    right_height = float(right_page.mediabox.height)

    new_width = left_width + right_width
    new_height = max(left_height, right_height)

    writer = PdfWriter()
    writer.add_blank_page(width=new_width, height=new_height)
    canvas = writer.pages[-1]
    canvas.mergeTranslatedPage(left_page, 0, new_height - left_height)
    canvas.mergeTranslatedPage(right_page, left_width, new_height - right_height)

    if labels:
        x_offset, y_offset = label_offset
        panel_offsets = [0.0, left_width]
        for label, x_base in zip(labels, panel_offsets):
            if not label:
                continue
            overlay = _create_label_overlay(label, new_width, new_height, x_base + x_offset, y_offset, label_font_size)
            canvas.merge_page(overlay)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output_file:
        writer.write(output_file)


def apply_label_to_pdf(
    source: Path,
    destination: Path,
    label: Optional[str] = None,
    label_offset: Tuple[float, float] = (36.0, 36.0),
    label_font_size: int = 48,
) -> None:
    """Copy `source` to `destination`, overlaying an optional panel label."""
    from PyPDF2 import PdfReader, PdfWriter

    if not source.exists():
        raise FileNotFoundError(f"Expected source PDF at {source}.")

    reader = PdfReader(str(source))
    if not reader.pages:
        raise ValueError("Source PDF must contain at least one page.")

    page = reader.pages[0]
    width = float(page.mediabox.width)
    height = float(page.mediabox.height)

    if label:
        x_offset, y_offset = label_offset
        overlay = _create_label_overlay(label, width, height, x_offset, y_offset, label_font_size)
        page.merge_page(overlay)

    writer = PdfWriter()
    writer.add_page(page)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output_file:
        writer.write(output_file)
