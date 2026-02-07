"""Utility helpers shared across manuscript figure drivers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader, PdfWriter


def _create_label_overlay(
    text: str,
    page_width: float,
    page_height: float,
    x_offset: float,
    y_offset: float,
    font_size: int,
) -> "PageObject":
    """Return a transparent PDF page containing a single label."""
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
