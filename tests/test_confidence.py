"""Confidence scoring: agreement, format checks, cross-field checks, review routing."""

from __future__ import annotations

import pytest

from pdf_extractor.confidence import (
    DEFAULT_CONFIG,
    cross_field_checks,
    review_queue,
    score_document,
    score_field,
    validate_format,
    values_agree,
)
from pdf_extractor.schemas import FIELD_SPECS, ExtractedValue


def ev(field: str, value, method: str = "rule_table", verified: bool | None = None) -> ExtractedValue:
    return ExtractedValue(
        field=field, value=value, raw=str(value), method=method, page=1, evidence_verified=verified
    )


TDC = FIELD_SPECS["total_dev_cost"]


def test_values_agree_numeric_tolerance_and_text_casefold() -> None:
    assert values_agree("money", 10_000_000, 10_010_000, rel_tol=0.005)
    assert not values_agree("money", 10_000_000, 10_100_000, rel_tol=0.005)
    assert values_agree("text", "Example  Housing LP", "example housing lp")
    assert not values_agree("text", None, "x")


def test_agreement_scores_highest() -> None:
    res = score_field(TDC, [ev("total_dev_cost", 1e7)], [ev("total_dev_cost", 1e7, "llm", True)])
    assert res.confidence == pytest.approx(DEFAULT_CONFIG.agree)
    assert res.status == "ok"
    assert "rule_llm_agree" in res.reasons


def test_disagreement_routes_to_review_and_keeps_rule_value() -> None:
    res = score_field(TDC, [ev("total_dev_cost", 1e7)], [ev("total_dev_cost", 2e7, "llm", True)])
    assert res.value == 1e7
    assert res.status == "review"


def test_llm_only_with_unverified_evidence_is_flagged() -> None:
    res = score_field(TDC, [], [ev("total_dev_cost", 1e7, "llm", False)])
    assert res.confidence == pytest.approx(DEFAULT_CONFIG.llm_only_unverified)
    assert res.status == "review"


def test_missing_field() -> None:
    res = score_field(TDC, [], [])
    assert res.status == "missing"
    assert res.value is None
    assert res.confidence == 0.0


def test_conflicting_rule_candidates_are_penalised() -> None:
    res = score_field(
        TDC,
        [ev("total_dev_cost", 1e7), ev("total_dev_cost", 1e7, "rule_regex"), ev("total_dev_cost", 9e6)],
        [],
    )
    assert res.value == 1e7  # majority value
    assert "rule_candidates_conflict" in res.reasons
    assert res.confidence == pytest.approx(DEFAULT_CONFIG.rule_table_only - DEFAULT_CONFIG.rule_conflict_penalty)


def test_format_validation() -> None:
    assert validate_format(TDC, -5.0) == ["money_not_positive"]
    assert validate_format(FIELD_SPECS["leverage_ratio"], 500.0) == ["ratio_out_of_range"]
    assert validate_format(FIELD_SPECS["project_id"], "ABC") == ["id_without_digit"]
    assert validate_format(TDC, 1e7) == []


def test_cross_field_leverage_and_hard_cost_checks() -> None:
    ok = {"total_dev_cost": 1e7, "funding_amount": 8e6, "leverage_ratio": 1.25, "hard_construction_cost": 6e6}
    assert cross_field_checks(ok) == {}

    bad = {**ok, "leverage_ratio": 1.60, "hard_construction_cost": 1.2e7}
    problems = cross_field_checks(bad)
    assert any(p.startswith("leverage_mismatch") for p in problems["leverage_ratio"])
    assert "hard_cost_exceeds_total" in problems["hard_construction_cost"]


def test_leverage_check_skipped_without_funding_amount() -> None:
    assert cross_field_checks({"total_dev_cost": 1e7, "leverage_ratio": 9.0}) == {}


def test_score_document_applies_cross_field_penalty_and_queue() -> None:
    rule = {
        "total_dev_cost": [ev("total_dev_cost", 1e7)],
        "funding_amount": [ev("funding_amount", 8e6)],
        "leverage_ratio": [ev("leverage_ratio", 1.60)],
    }
    llm = {"total_dev_cost": [ev("total_dev_cost", 1e7, "llm", True)]}
    doc = score_document("x.pdf", rule, llm)
    lev = doc.fields["leverage_ratio"]
    assert lev.status == "review"
    assert lev.confidence == pytest.approx(DEFAULT_CONFIG.rule_table_only - DEFAULT_CONFIG.cross_field_penalty)
    assert doc.needs_review
    assert doc.data_confidence_label == "Review Needed"
    queued = [res.field for _, res in review_queue([doc])]
    assert "leverage_ratio" in queued
    assert "applicant_name" in queued  # required and missing
    assert "funding_amount" in queued  # implicated by the leverage mismatch
