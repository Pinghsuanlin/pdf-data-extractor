"""End-to-end on a generated PDF, plus export and human edits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pdf_extractor.export import (
    apply_human_edits,
    audit_frame,
    display_frame,
    export_results,
    records_frame,
)
from pdf_extractor.llm_extract import FakeLLM
from pdf_extractor.parse import ParsedDocument, locate_snippet, parse_pdf
from pdf_extractor.pipeline import PipelineConfig, extract_document, extract_parsed


def test_parse_pdf_reads_text_tables_and_boxes(sample_pdf: Path) -> None:
    doc = parse_pdf(sample_pdf, ocr="off")
    assert len(doc.pages) == 2
    assert "Example Housing LP" in doc.pages[0].text
    assert doc.pages[0].tables and doc.pages[0].tables[0].rows[0][0] == "Total Development Cost"
    assert locate_snippet(doc.pages[0], "Example Housing LP") is not None


def test_end_to_end_rules_only(sample_pdf: Path) -> None:
    result = extract_document(sample_pdf, backend=None, config=PipelineConfig(llm_policy="off", ocr="off"))
    values = result.values()
    assert values["project_id"] == "2099-001T"
    assert values["total_dev_cost"] == pytest.approx(10_000_000)
    assert values["leverage_ratio"] == pytest.approx(1.25)
    assert result.fields["total_dev_cost"].chosen.method == "rule_table"
    assert result.fields["leverage_ratio"].chosen.page == 2
    assert not result.needs_review


def test_end_to_end_with_disagreeing_llm(sample_doc: ParsedDocument) -> None:
    fake = FakeLLM({"fields": {"applicant_name": {"value": "Someone Else LLC", "page": 1}}})
    result = extract_parsed(sample_doc, fake, PipelineConfig(llm_policy="hard"))
    assert result.fields["applicant_name"].status == "review"
    assert result.data_confidence_label == "Review Needed"


def test_exports_and_audit_log(sample_doc: ParsedDocument, tmp_path: Path) -> None:
    result = extract_parsed(sample_doc, None, PipelineConfig(llm_policy="off"))
    paths = export_results([result], tmp_path)
    assert {p.name for p in paths} == {"records.csv", "review_queue.csv", "audit_log.csv", "extraction.xlsx"}
    records = pd.read_csv(tmp_path / "records.csv")
    assert records.loc[0, "total_dev_cost"] == pytest.approx(10_000_000)
    audit = audit_frame([result])
    assert {"page", "x0", "top", "x1", "bottom", "method", "reasons"} <= set(audit.columns)
    assert audit.set_index("field").loc["leverage_ratio", "page"] == 1
    sheets = pd.read_excel(tmp_path / "extraction.xlsx", sheet_name=None)
    assert set(sheets) == {"records", "review_queue", "audit_log"}


def test_human_edit_is_recorded(sample_doc: ParsedDocument) -> None:
    result = extract_parsed(sample_doc, None, PipelineConfig(llm_policy="off"))
    result.fields["applicant_name"].status = "review"
    apply_human_edits(result, {"applicant_name": "Example Housing LP (verified)"})
    res = result.fields["applicant_name"]
    assert res.status == "ok" and res.confidence == 1.0
    assert res.previous_value == "Example Housing LP"
    assert res.chosen.method == "human"
    assert records_frame([result]).loc[0, "applicant_name"] == "Example Housing LP (verified)"


def test_display_frame_uses_markers() -> None:
    empty = extract_parsed(ParsedDocument(path="blank.pdf", pages=[]), None)
    row = display_frame([empty]).iloc[0]
    assert row["applicant_name"] == "[Missing]"
    assert row["data_confidence"] == "Review Needed"
