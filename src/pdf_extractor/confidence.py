"""Per-field confidence scoring and the human-review decision.

The score is a transparent rule table, not a learned model, so a reviewer can read why a field
was flagged. It combines three signals:

1. **Source agreement**: do the rule and the LLM give the same value? A rule value read from OCR
   text (a scanned page) is not trusted on its own: OCR misreads characters ('210S' as '2108')
   in ways no format check catches, so it needs the LLM to agree before it skips review.
2. **Format validity**: is the value plausible for its type (positive dollars, sane ratio, id with
   a digit)?
3. **Cross-field checks**: hard construction cost should not exceed total development cost, and
   the stated leverage ratio should be close to total development cost / funding amount when both
   are present.

The numbers in ``ConfidenceConfig`` are hand-set starting points. ``evaluate.py`` reports whether
flagged fields really are wrong more often; use that to recalibrate them on labelled documents.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional

from pdf_extractor.schemas import (
    FIELD_SPECS,
    DocumentResult,
    ExtractedValue,
    FieldResult,
    FieldSpec,
)


@dataclass(frozen=True)
class ConfidenceConfig:
    """Scores and tolerances used by the scorer. All scores are in [0, 1]."""

    agree: float = 0.95
    rule_table_only: float = 0.85
    rule_regex_only: float = 0.75
    llm_only: float = 0.75
    llm_only_fuzzy_evidence: float = 0.60
    llm_only_unverified: float = 0.40
    disagree: float = 0.35
    rule_conflict_penalty: float = 0.20
    multi_row_penalty: float = 0.10
    ocr_penalty: float = 0.20
    invalid_format_factor: float = 0.50
    cross_field_penalty: float = 0.30
    review_threshold: float = 0.70
    numeric_rel_tol: float = 0.005
    leverage_rel_tol: float = 0.05
    min_money: float = 1_000.0
    max_money: float = 2e9
    max_ratio: float = 50.0


DEFAULT_CONFIG = ConfidenceConfig()


def values_agree(kind: str, a: Any, b: Any, rel_tol: float = DEFAULT_CONFIG.numeric_rel_tol) -> bool:
    """Compare two normalised values: relative tolerance for numbers, casefold for text."""
    if a is None or b is None:
        return False
    if kind in {"money", "ratio"}:
        try:
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return " ".join(str(a).split()).casefold() == " ".join(str(b).split()).casefold()


def validate_format(spec: FieldSpec, value: Any, cfg: ConfidenceConfig = DEFAULT_CONFIG) -> list[str]:
    """Return a list of format problems for one value (empty when it looks valid)."""
    if value is None:
        return []
    problems: list[str] = []
    if spec.kind == "money":
        if not isinstance(value, (int, float)):
            problems.append("money_not_numeric")
        elif value <= 0:
            problems.append("money_not_positive")
        elif value < cfg.min_money:
            # Development budgets are never a few dollars; this catches OCR splits like '$1 7,775,000'.
            problems.append("money_implausibly_small")
        elif value > cfg.max_money:
            problems.append("money_implausibly_large")
    elif spec.kind == "ratio":
        if not isinstance(value, (int, float)):
            problems.append("ratio_not_numeric")
        elif not 0 < value <= cfg.max_ratio:
            problems.append("ratio_out_of_range")
    elif spec.kind == "id":
        if not any(ch.isdigit() for ch in str(value)):
            problems.append("id_without_digit")
    elif spec.kind == "text":
        text = str(value)
        if len(text) < 2 or len(text) > 150:
            problems.append("text_length_implausible")
        if not any(ch.isalpha() for ch in text):
            # A bare number captured as a name. Digit-heavy text such as 'RFA 2023-102' is fine.
            problems.append("text_has_no_letters")
    return problems


def cross_field_checks(
    values: dict[str, Any], cfg: ConfidenceConfig = DEFAULT_CONFIG
) -> dict[str, list[str]]:
    """Check consistency between fields. Returns field name to problems found.

    Checks run only when the inputs are present, so a missing funding amount never penalises the
    leverage ratio.
    """
    problems: dict[str, list[str]] = {}

    def add(fields: tuple[str, ...], reason: str) -> None:
        for name in fields:
            problems.setdefault(name, []).append(reason)

    tdc = values.get("total_dev_cost")
    hard = values.get("hard_construction_cost")
    funding = values.get("funding_amount")
    leverage = values.get("leverage_ratio")

    if isinstance(tdc, (int, float)) and isinstance(hard, (int, float)) and hard > tdc:
        add(("total_dev_cost", "hard_construction_cost"), "hard_cost_exceeds_total")

    if (
        isinstance(tdc, (int, float))
        and isinstance(funding, (int, float))
        and isinstance(leverage, (int, float))
        and funding > 0
    ):
        implied = tdc / funding
        if not math.isclose(implied, leverage, rel_tol=cfg.leverage_rel_tol):
            add(
                ("leverage_ratio", "funding_amount", "total_dev_cost"),
                f"leverage_mismatch(implied={implied:.2f})",
            )
    return problems


def _pick_rule_value(
    spec: FieldSpec, rule: list[ExtractedValue], cfg: ConfidenceConfig
) -> tuple[Optional[ExtractedValue], bool]:
    """Choose the rule candidate to trust and report whether rules disagreed with each other.

    The most common value wins; ties go to table hits (they come first in the list).
    """
    if not rule:
        return None, False
    groups: list[list[ExtractedValue]] = []
    for cand in rule:
        for group in groups:
            if values_agree(spec.kind, group[0].value, cand.value, cfg.numeric_rel_tol):
                group.append(cand)
                break
        else:
            groups.append([cand])
    counts = Counter({i: len(g) for i, g in enumerate(groups)})
    best_idx = max(counts, key=lambda i: (counts[i], -i))
    return groups[best_idx][0], len(groups) > 1


def score_field(
    spec: FieldSpec,
    rule: list[ExtractedValue],
    llm: list[ExtractedValue],
    cfg: ConfidenceConfig = DEFAULT_CONFIG,
) -> FieldResult:
    """Score one field from its rule and LLM candidates (before cross-field checks)."""
    reasons: list[str] = []
    chosen_rule, rule_conflict = _pick_rule_value(spec, rule, cfg)
    llm_cand = next((c for c in llm if c.value is not None), None)
    candidates = [*rule, *llm]

    if llm and llm_cand is None:
        reasons.append("llm_value_unparseable")

    if chosen_rule is not None and llm_cand is not None:
        if values_agree(spec.kind, chosen_rule.value, llm_cand.value, cfg.numeric_rel_tol):
            chosen, score = chosen_rule, cfg.agree
            reasons.append("rule_llm_agree")
        else:
            # Keep the rule value unless it fails a format check the LLM value passes (e.g. OCR read
            # '$1 7,775,000' as $1). Either way the field goes to review.
            rule_bad = bool(validate_format(spec, chosen_rule.value, cfg))
            llm_ok = not validate_format(spec, llm_cand.value, cfg)
            chosen = llm_cand if rule_bad and llm_ok else chosen_rule
            score = cfg.disagree
            other = chosen_rule.value if chosen is llm_cand else llm_cand.value
            source = "rule" if chosen is llm_cand else "llm"
            reasons.append(f"rule_llm_disagree({source}={other!r})")
    elif chosen_rule is not None:
        chosen = chosen_rule
        if chosen_rule.method == "rule_table":
            score = cfg.rule_table_only
            reasons.append("rule_table_only")
        else:
            score = cfg.rule_regex_only
            reasons.append("rule_regex_only")
    elif llm_cand is not None:
        chosen = llm_cand
        if llm_cand.evidence_verified and "evidence_fuzzy_ocr" in llm_cand.notes:
            score = cfg.llm_only_fuzzy_evidence
            reasons.append("llm_only_evidence_fuzzy_ocr")
        elif llm_cand.evidence_verified:
            score = cfg.llm_only
            reasons.append("llm_only")
        else:
            score = cfg.llm_only_unverified
            reasons.append("llm_only_evidence_not_found")
    else:
        return FieldResult(
            field=spec.name,
            value=None,
            confidence=0.0,
            status="missing",
            reasons=["not_found"],
            candidates=candidates,
        )

    if rule_conflict:
        score -= cfg.rule_conflict_penalty
        reasons.append("rule_candidates_conflict")
    if "from_ocr" in chosen.notes and "rule_llm_agree" not in reasons:
        score -= cfg.ocr_penalty
        reasons.append("ocr_unconfirmed")
    if "multi_row_table" in chosen.notes:
        score -= cfg.multi_row_penalty
        reasons.append("multi_row_table")
    format_problems = validate_format(spec, chosen.value, cfg)
    if format_problems:
        score *= cfg.invalid_format_factor
        reasons.extend(format_problems)
    if "read_from_image" in chosen.notes:
        reasons.append("read_from_image")

    score = max(0.0, min(1.0, score))
    return FieldResult(
        field=spec.name,
        value=chosen.value,
        confidence=score,
        status="ok" if score >= cfg.review_threshold else "review",
        reasons=reasons,
        chosen=chosen,
        candidates=candidates,
    )


def score_document(
    source_file: str,
    rule_candidates: dict[str, list[ExtractedValue]],
    llm_candidates: dict[str, list[ExtractedValue]],
    cfg: ConfidenceConfig = DEFAULT_CONFIG,
    warnings: Optional[list[str]] = None,
) -> DocumentResult:
    """Score every field, apply cross-field checks and set review status."""
    results = {
        name: score_field(spec, rule_candidates.get(name, []), llm_candidates.get(name, []), cfg)
        for name, spec in FIELD_SPECS.items()
    }
    problems = cross_field_checks({n: r.value for n, r in results.items()}, cfg)
    for name, reasons in problems.items():
        result = results[name]
        if result.value is None:
            continue
        result.confidence = max(0.0, result.confidence - cfg.cross_field_penalty)
        result.reasons.extend(reasons)
        result.status = "ok" if result.confidence >= cfg.review_threshold else "review"
    return DocumentResult(source_file=source_file, fields=results, warnings=list(warnings or []))


def review_queue(results: list[DocumentResult]) -> list[tuple[DocumentResult, FieldResult]]:
    """All fields needing a human, lowest confidence first. Missing optional fields are skipped."""
    queue = [
        (doc, res)
        for doc in results
        for name, res in doc.fields.items()
        if res.status == "review" or (res.status == "missing" and FIELD_SPECS[name].required)
    ]
    return sorted(queue, key=lambda item: item[1].confidence)
