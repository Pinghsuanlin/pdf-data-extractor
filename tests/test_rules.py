"""Normalisers and rule extraction on text fixtures."""

from __future__ import annotations

import pytest

from pdf_extractor.parse import ParsedDocument, page_from_text
from pdf_extractor.rules import (
    extract_from_table,
    extract_rules,
    match_label,
    parse_money,
    parse_ratio,
)
from pdf_extractor.schemas import FIELD_SPECS


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("$14,500,000", 14_500_000.0),
        ("14,500,000.00", 14_500_000.0),
        ("$9.2M", 9_200_000.0),
        ("14.5 million", 14_500_000.0),
        ("$250k", 250_000.0),
        ("(1,000)", -1_000.0),
        ("US$ 3,000", 3_000.0),
    ],
)
def test_parse_money_variants(raw: str, expected: float) -> None:
    assert parse_money(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["", "n/a", "see attached", "$", "12/31/2023"])
def test_parse_money_rejects_non_amounts(raw: str) -> None:
    assert parse_money(raw) is None


@pytest.mark.parametrize(
    ("raw", "expected"), [("1.45", 1.45), ("1.45x", 1.45), ("1.45:1", 1.45), ("145%", 1.45)]
)
def test_parse_ratio_variants(raw: str, expected: float) -> None:
    assert parse_ratio(raw) == pytest.approx(expected)


def test_match_label_is_exact_after_normalising() -> None:
    specs = list(FIELD_SPECS.values())
    assert match_label("Total Development Cost:", specs).name == "total_dev_cost"
    assert match_label("3. Total Development Cost ($)", specs).name == "total_dev_cost"
    assert match_label("Total Development Cost per Unit", specs) is None


def test_extract_rules_from_text(sample_doc: ParsedDocument) -> None:
    found = extract_rules(sample_doc)
    first = {name: vals[0].value for name, vals in found.items() if vals}
    assert first["project_id"] == "2099-001T"
    assert first["applicant_name"] == "Example Housing LP"
    assert first["funding_source"] == "RFA 2099-100"
    assert first["total_dev_cost"] == pytest.approx(10_000_000)
    assert first["hard_construction_cost"] == pytest.approx(6_500_000)
    assert first["funding_amount"] == pytest.approx(8_000_000)
    assert first["leverage_ratio"] == pytest.approx(1.25)


def test_table_hit_wins_over_duplicate_regex_hit(sample_doc: ParsedDocument) -> None:
    found = extract_rules(sample_doc)
    tdc = found["total_dev_cost"]
    assert [v.method for v in tdc] == ["rule_table"]


def test_prose_does_not_trigger_text_fields() -> None:
    page = page_from_text(1, "The applicant: shall provide evidence of site control.\n")
    doc = ParsedDocument(path="x.pdf", pages=[page])
    # 'The applicant:' does not start the line, so no applicant_name is extracted.
    assert extract_rules(doc)["applicant_name"] == []


def test_header_table_layout_and_multi_row_note() -> None:
    table = [
        ["Application Number", "Total Development Cost", "Leverage Ratio"],
        ["2099-002A", "$12,000,000", "1.35"],
        ["2099-003B", "$9,000,000", "1.20"],
    ]
    page = page_from_text(1, "", [table])
    values = extract_from_table(page, page.tables[0], FIELD_SPECS.values())
    by_field = {v.field: v for v in values}
    assert by_field["project_id"].value == "2099-002A"
    assert by_field["total_dev_cost"].value == pytest.approx(12_000_000)
    assert "multi_row_table" in by_field["leverage_ratio"].notes


def test_regex_hit_has_page_and_bbox(sample_doc: ParsedDocument) -> None:
    hit = extract_rules(sample_doc)["leverage_ratio"][0]
    assert hit.page == 1
    assert hit.bbox is not None


def test_header_row_with_one_known_label_is_still_a_header() -> None:
    from pdf_extractor.parse import page_from_text
    from pdf_extractor.rules import extract_from_table
    from pdf_extractor.schemas import FIELD_SPECS

    rows = [["Submission Reference", "Funding Program", "Agency Request"],
            ["2024-333BS", "RFA 2024-138", "$11,610,000"]]
    page = page_from_text(1, "", [rows])
    found = extract_from_table(page, page.tables[0], FIELD_SPECS.values())
    assert [(v.field, v.value) for v in found] == [("funding_source", "RFA 2024-138")]
