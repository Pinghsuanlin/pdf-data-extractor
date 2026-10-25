"""Evaluation harness: gold loading (FILL_IN skipping) and metric arithmetic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pdf_extractor.confidence import score_document
from pdf_extractor.evaluate import GoldDocument, evaluate, load_gold, metrics_markdown
from pdf_extractor.schemas import ExtractedValue

TEMPLATE = Path(__file__).resolve().parents[1] / "eval" / "gold" / "TEMPLATE_application.json"


def ev(field: str, value) -> ExtractedValue:
    return ExtractedValue(field=field, value=value, raw=str(value), method="rule_table", page=1)


def test_shipped_template_is_skipped_as_unlabelled() -> None:
    assert TEMPLATE.exists()
    assert load_gold(TEMPLATE.parent) == []


def test_load_gold_skips_fill_in_and_normalises(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text(
        json.dumps(
            {
                "source_file": "a.pdf",
                "fields": {"total_dev_cost": "$1,000,000", "applicant_name": "FILL_IN", "funding_amount": None},
            }
        )
    )
    gold = load_gold(tmp_path)
    assert len(gold) == 1
    assert gold[0].fields == {"total_dev_cost": 1_000_000.0, "funding_amount": None}


def test_metrics_including_flag_precision() -> None:
    # One correct confident value, one wrong flagged value, one wrong confident value,
    # one correct "absent" field, one missed field.
    doc = score_document(
        "a.pdf",
        {
            "total_dev_cost": [ev("total_dev_cost", 1e7)],
            "hard_construction_cost": [ev("hard_construction_cost", 5e6)],
            "project_id": [ev("project_id", "2099-9")],
        },
        {},
    )
    doc.fields["hard_construction_cost"].status = "review"
    gold = [
        GoldDocument(
            "a.pdf",
            {
                "total_dev_cost": 1e7,
                "hard_construction_cost": 6e6,
                "project_id": "2099-1",
                "funding_amount": None,
                "applicant_name": "Example Housing LP",
            },
        )
    ]
    m = evaluate([doc], gold)
    assert m["n_labelled_fields"] == 5
    assert m["accuracy"] == pytest.approx(2 / 5)
    assert m["coverage"] == pytest.approx(3 / 4)
    assert m["flag_precision"] == pytest.approx(1.0)
    assert m["unflagged_error_rate"] == pytest.approx(1 / 2)
    assert m["flag_recall"] == pytest.approx(1 / 2)
    assert "| Field accuracy |" in metrics_markdown(m)
