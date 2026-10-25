"""Layout-aware PDF parsing with pdfplumber, plus an optional OCR fallback.

The parser keeps three views of every page, because different extractors need different ones:

* ``text``: reading-order text, used by the regex rules and as LLM context.
* ``words``: every word with its bounding box, used to locate a value on the page for the audit log.
* ``tables``: cell grids with cell bounding boxes, used by the table-header rules.

Scanned pages have no text layer. When ``ocr`` is enabled and a page has (almost) no text,
the page is rendered to an image and passed to Tesseract via ``pytesseract``. OCR is an optional
extra (``pip install -e ".[ocr]"`` plus the Tesseract binary) and is not exercised in CI.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from pdf_extractor.schemas import BBox

logger = logging.getLogger(__name__)

OcrMode = Literal["off", "auto", "force"]
OCR_RESOLUTION = 300  # dpi used when rendering a page for OCR


@dataclass
class Word:
    """A single word and its position on the page."""

    text: str
    bbox: BBox


@dataclass
class ParsedTable:
    """A table as a grid of cell strings, with optional cell bounding boxes."""

    rows: list[list[str]]
    bbox: Optional[BBox] = None
    cell_bboxes: Optional[list[list[Optional[BBox]]]] = None

    def cell_bbox(self, row: int, col: int) -> Optional[BBox]:
        """Bounding box of one cell, or the table bbox when cell geometry is unknown."""
        if self.cell_bboxes is not None:
            try:
                box = self.cell_bboxes[row][col]
            except IndexError:
                box = None
            if box is not None:
                return box
        return self.bbox


@dataclass
class ParsedPage:
    """One page of a parsed document. ``number`` is 1-based to match PDF viewers."""

    number: int
    text: str
    words: list[Word] = field(default_factory=list)
    tables: list[ParsedTable] = field(default_factory=list)
    ocr: bool = False


@dataclass
class ParsedDocument:
    """A parsed PDF: its path and pages."""

    path: str
    pages: list[ParsedPage]

    @property
    def full_text(self) -> str:
        """All page text joined with page markers (useful for debugging)."""
        return "\n".join(f"[Page {p.number}]\n{p.text}" for p in self.pages)


def _clean_cell(cell: Optional[str]) -> str:
    """Collapse whitespace inside a table cell; ``None`` becomes an empty string."""
    if cell is None:
        return ""
    return re.sub(r"\s+", " ", cell).strip()


def _parse_tables(page: object) -> list[ParsedTable]:
    """Extract tables from a pdfplumber page, keeping per-cell bounding boxes."""
    tables: list[ParsedTable] = []
    for table in page.find_tables():  # type: ignore[attr-defined]
        grid = [[_clean_cell(c) for c in row] for row in table.extract()]
        cell_boxes: list[list[Optional[BBox]]] = []
        for row in table.rows:
            cell_boxes.append([tuple(c) if c is not None else None for c in row.cells])  # type: ignore[misc]
        tables.append(ParsedTable(rows=grid, bbox=tuple(table.bbox), cell_bboxes=cell_boxes))  # type: ignore[arg-type]
    return tables


def _ocr_page(page: object) -> tuple[str, list[Word]]:
    """OCR one pdfplumber page. Raises ImportError when the OCR extra is not installed."""
    import pytesseract  # optional dependency

    image = page.to_image(resolution=OCR_RESOLUTION).original  # type: ignore[attr-defined]
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    scale = 72.0 / OCR_RESOLUTION  # image pixels back to PDF points
    words: list[Word] = []
    lines: dict[tuple[int, int, int], list[str]] = {}
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        words.append(Word(text=text, bbox=(x * scale, y * scale, (x + w) * scale, (y + h) * scale)))
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(text)
    return "\n".join(" ".join(ws) for ws in lines.values()), words


def parse_pdf(path: str | Path, ocr: OcrMode = "auto", min_chars: int = 20) -> ParsedDocument:
    """Parse a PDF into text, word boxes and tables for each page.

    Args:
        path: Path to the PDF.
        ocr: ``"off"`` never runs OCR; ``"auto"`` runs it on pages with fewer than ``min_chars``
            characters of text; ``"force"`` runs it on every page.
        min_chars: Threshold for treating a page as scanned in ``"auto"`` mode.

    Returns:
        A ``ParsedDocument``. Pages that need OCR but cannot get it are kept with empty text and a
        logged warning, so one scanned page never stops the whole document.
    """
    import pdfplumber

    pages: list[ParsedPage] = []
    with pdfplumber.open(str(path)) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            words = [
                Word(text=w["text"], bbox=(w["x0"], w["top"], w["x1"], w["bottom"]))
                for w in page.extract_words()
            ]
            tables = _parse_tables(page)
            used_ocr = False
            needs_ocr = ocr == "force" or (ocr == "auto" and len(text.strip()) < min_chars)
            if needs_ocr:
                try:
                    text, words = _ocr_page(page)
                    used_ocr = True
                except ImportError:
                    logger.warning(
                        "Page %d of %s looks scanned but OCR extra is not installed.", index, path
                    )
                except Exception as exc:  # Tesseract binary missing, render failure, etc.
                    logger.warning("OCR failed on page %d of %s: %s", index, path, exc)
            pages.append(
                ParsedPage(number=index, text=text, words=words, tables=tables, ocr=used_ocr)
            )
    return ParsedDocument(path=str(path), pages=pages)


def page_from_text(number: int, text: str, tables: Optional[list[list[list[str]]]] = None) -> ParsedPage:
    """Build a ``ParsedPage`` from plain text, for tests and non-PDF inputs.

    Words get synthetic, line-based bounding boxes so that ``locate_snippet`` still works.
    """
    words: list[Word] = []
    for line_no, line in enumerate(text.splitlines()):
        x = 0.0
        for token in line.split():
            width = 6.0 * len(token)
            words.append(Word(text=token, bbox=(x, 12.0 * line_no, x + width, 12.0 * line_no + 10)))
            x += width + 4.0
    parsed_tables = [ParsedTable(rows=t) for t in (tables or [])]
    return ParsedPage(number=number, text=text, words=words, tables=parsed_tables)


def _norm_token(token: str) -> str:
    """Lower-case a token and strip surrounding punctuation for fuzzy word matching."""
    return token.lower().strip(".,;:()[]{}\"'")


def locate_snippet(page: ParsedPage, snippet: str) -> Optional[BBox]:
    """Find the bounding box of ``snippet`` on a page by matching its word sequence.

    Matching is case-insensitive and ignores surrounding punctuation. Returns the union of the
    matched word boxes, or ``None`` when the snippet is not on the page.
    """
    target = [_norm_token(t) for t in snippet.split() if _norm_token(t)]
    if not target or not page.words:
        return None
    tokens = [_norm_token(w.text) for w in page.words]
    n = len(target)
    for start in range(len(tokens) - n + 1):
        if tokens[start : start + n] == target:
            boxes = [page.words[i].bbox for i in range(start, start + n)]
            return (
                min(b[0] for b in boxes),
                min(b[1] for b in boxes),
                max(b[2] for b in boxes),
                max(b[3] for b in boxes),
            )
    return None
