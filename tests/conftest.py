"""Shared fixtures. Every fixture is synthetic and built in the test; no real documents ship."""

from __future__ import annotations

from pathlib import Path

import pytest

from pdf_extractor.parse import ParsedDocument, page_from_text

SAMPLE_TEXT = """Example Funding Application (synthetic test fixture)
Application Number: 2099-001T
Name of Applicant: Example Housing LP
Funding Source: RFA 2099-100
Total Development Cost: $10,000,000
Hard Construction Cost: $6,500,000
Requested Amount: $8,000,000
Leverage Ratio: 1.25
"""

SAMPLE_TABLE = [
    ["Total Development Cost", "$10,000,000"],
    ["Hard Construction Cost", "$6,500,000"],
]


@pytest.fixture
def sample_doc() -> ParsedDocument:
    """A one-page parsed document built from text (no PDF needed)."""
    return ParsedDocument(path="synthetic.pdf", pages=[page_from_text(1, SAMPLE_TEXT, [SAMPLE_TABLE])])


def build_pdf(path: Path) -> Path:
    """Write a tiny two-page PDF with a key-value table, using reportlab."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle

    styles = getSampleStyleSheet()
    table = Table(SAMPLE_TABLE)
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black)]))
    story = [
        Paragraph("Application Number: 2099-001T", styles["Normal"]),
        Paragraph("Name of Applicant: Example Housing LP", styles["Normal"]),
        Paragraph("Funding Source: RFA 2099-100", styles["Normal"]),
        table,
        PageBreak(),
        Paragraph("Requested Amount: $8,000,000", styles["Normal"]),
        Paragraph("Leverage Ratio: 1.25", styles["Normal"]),
    ]
    SimpleDocTemplate(str(path), pagesize=letter).build(story)
    return path


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Path to a generated PDF; skips the test when reportlab is not installed."""
    pytest.importorskip("reportlab")
    return build_pdf(tmp_path / "synthetic.pdf")
